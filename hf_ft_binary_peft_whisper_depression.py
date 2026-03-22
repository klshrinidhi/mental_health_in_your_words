import sys
import os
from collections import defaultdict
import argparse
import shutil
import pathlib
import json

from tqdm import tqdm
import wandb
import numpy as np
import sklearn.metrics as skm
import torchaudio
from transformers import (AutoFeatureExtractor,
                          AutoModelForAudioClassification,
                          TrainingArguments,
                          Trainer,
                          logging)
from peft import LoraConfig,TaskType,get_peft_model
import datasets

# Make sure to run this script in a while loop as the Trainer's dataloader 
# sometimes fails after several mins of training.

## References:
## https://huggingface.co/blog/fine-tune-wav2vec2-english
## https://huggingface.co/docs/transformers/main/en/tasks/audio_classification
## https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb

# logging.set_verbosity_info()

label_to_id = {'minimal':0,'noticeable':1}
id_to_label = {v:k for k,v in label_to_id.items()}

argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu',type=int,default=0)
argparser.add_argument('--lora_rank',type=int,default=8)
argparser.add_argument('--i_beg',type=int,default=0)
argparser.add_argument('--i_end',type=int,default=15)
args = argparser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

epochs = 10
batch_size = 8
learning_rate = 1e-3
lora_rank = args.lora_rank
sampling_rate = 16000
audio_chunk_dur = 30 # sec
audio_chunk_overlap_dur = 5 # sec
run_name = f'whisper_tiny_bin_peft_dep_e{epochs}_bs{batch_size}_lr{learning_rate}_r{lora_rank}'
model_name = 'openai/whisper-tiny.en'
root_d = pathlib.Path('ROOT_D') / sys.argv[1]
prompts_user_d = root_d / 'prompts_user'
train_val_d = root_d / 'train_val'
run_output_d = root_d / run_name
run_output_d.mkdir(parents=True,exist_ok=True)

train_fs = list(sorted(train_val_d.glob('train_dep_*.jsonl')))
val_fs = list(sorted(train_val_d.glob('val_dep_*.jsonl')))
assert len(train_fs) == len(val_fs) == 15

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
def tokenize(examples):
    # HuggingFace Trainer expects keys 'label' (integer) and all the keys in 
    # the tokenizer output -- 'input_features','attention_mask','input_type_ids'.
    result = {'label':[(0 if l == 'minimal' else 1) for l in examples['label']]}
    audio = [x['array'] for x in examples['audio']]
    result.update(feature_extractor(audio,sampling_rate=sampling_rate,padding="longest"))
    return result

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics= {'accuracy':skm.accuracy_score(labels,predictions),
              'balanced_accuracy':skm.balanced_accuracy_score(labels,predictions),
              'f1':skm.f1_score(labels,predictions,average='weighted'),
              'precision':skm.precision_score(labels,predictions,average='weighted'),
              'recall':skm.recall_score(labels,predictions,average='weighted')}
    return metrics

def create_dataset(f):
    lines = f.open().readlines()
    dataset = defaultdict(list)
    for line in tqdm(lines,desc=f'create dataset [{f.name}]',leave=False):
        example = json.loads(line.strip())
        # Split the audio into `input_audio_dur` sec chunks.
        audio_f = prompts_user_d / (example['participant_id'] + '.mp3')
        wf,sr = torchaudio.load(str(audio_f))
        wf = wf.mean(dim=0,keepdim=True)
        n_frames_per_chunk = int(sr * audio_chunk_dur)
        n_frames_step = int(sr * (audio_chunk_dur - audio_chunk_overlap_dur))
        assert wf.shape[1] >= n_frames_per_chunk
        # Drop few secs from the beginnning if we can't split into equal chunks.
        i_beg = (wf.shape[1] - n_frames_per_chunk) % n_frames_step
        wf = wf[:,i_beg:]
        n_chunks = ((wf.shape[1] - n_frames_per_chunk) // n_frames_step) + 1
        for i_chunk in range(n_chunks):
            audio_chunk_f = prompts_user_d / f'whisper_{example["participant_id"]}_{i_chunk:02}.wav'
            if not audio_chunk_f.is_file():
                i_beg = i_chunk * n_frames_step
                i_end = i_beg + n_frames_per_chunk
                wf_chunk = wf[:,i_beg:i_end]
                assert wf_chunk.shape[1] == n_frames_per_chunk
                torchaudio.save(str(audio_chunk_f),wf_chunk,sr)
            # dataset['participant_id'].append(example['participant_id'])
            # dataset['chunk'].append(i_chunk)
            dataset['label'].append(example['label'])
            dataset['audio'].append(str(audio_chunk_f))
    return dataset

def train_and_evaluate(train_f, val_f):
    output_d = run_output_d / train_f.stem.replace('train','').replace('_','')
    results_f = output_d / 'results.json'
    if results_f.is_file():
        print(f'{results_f} exists')
        return
    shutil.rmtree(output_d,ignore_errors=True)
    output_d.mkdir(parents=True,exist_ok=True)
    dataset_train = datasets.Dataset.from_dict(create_dataset(train_f),split='train')
    dataset_train = dataset_train.cast_column('audio',datasets.Audio(sampling_rate=sampling_rate))
    dataset_val = datasets.Dataset.from_dict(create_dataset(val_f),split='val')
    dataset_val = dataset_val.cast_column('audio',datasets.Audio(sampling_rate=sampling_rate))
    dataset = datasets.DatasetDict({'train':dataset_train,
                                    'val':dataset_val})
    # Without restricting batch_size, the tokenization step runs out of memory.
    tokenized_dataset = dataset.map(tokenize,remove_columns='audio',batched=True,batch_size=batch_size)
    model = AutoModelForAudioClassification.from_pretrained(model_name,
                                                            num_labels=len(label_to_id),
                                                            id2label=id_to_label,
                                                            label2id=label_to_id,
                                                            torch_dtype='auto')
    peft_config = LoraConfig(r=lora_rank,
                             target_modules='all-linear')
    model = get_peft_model(model,peft_config)
    model.print_trainable_parameters()
    model = model.cuda()
    # The default data collator in Training reads `label` column and outputs `labels` column when
    # creating a batch. We need to use labels_names=['labels'] to tell the Trainer to look for 'labels'
    # during evaluation. I found this out by tracing the Trainer code.
    training_args = TrainingArguments(output_dir=str(output_d),
                                      eval_strategy='epoch',
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      num_train_epochs=epochs,
                                      logging_steps=1,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      report_to='wandb',
                                      dataloader_num_workers=2,
                                      dataloader_prefetch_factor=2,
                                      run_name=f'{run_name}_{output_d.name}',
                                      label_names=['labels'])
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset, # tokenized_dataset['val'],
                      compute_metrics=compute_metrics)
    run = wandb.init(project='depression_anxiety',
                     name=f'{run_name}_{output_d.name}')
    trainer.train()
    results = trainer.evaluate(tokenized_dataset)
    model.print_trainable_parameters()
    json.dump(results,open(results_f,'w'),indent=2)
    run.finish()

pbar = tqdm(list(zip(train_fs,val_fs))[args.i_beg:args.i_end],desc=run_name)
for train_f,val_f in pbar:
    assert train_f.stem.replace('train','val') == val_f.stem
    pbar.set_postfix({'train':train_f.name,'val':val_f.name})
    train_and_evaluate(train_f,val_f)
