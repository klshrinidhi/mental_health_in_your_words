import sys
import os
import argparse
import pathlib
import json

from tqdm import tqdm
import numpy as np
import sklearn.metrics as skm
import torch
from torchvision.transforms import (Compose,
                                    Lambda,
                                    CenterCrop,
                                    RandomHorizontalFlip,
                                    Resize)
from transformers import (AutoFeatureExtractor,
                          AutoConfig,
                          AutoModelForVideoClassification,
                          TrainingArguments,
                          Trainer,
                          logging)
from pytorchvideo.data import (LabeledVideoDataset,
                               make_clip_sampler)
from pytorchvideo.transforms import (ApplyTransformToKey,
                                     Normalize,
                                     RandomShortSideScale,
                                     UniformTemporalSubsample)
# from peft import LoraConfig,TaskType,get_peft_model

## References:
## https://huggingface.co/docs/transformers/tasks/video_classification#train-the-model

logging.set_verbosity_info()

label_to_id = {'minimal':0,'mild':1,'moderate':2,'severe':3}
id_to_label = {v:k for k,v in label_to_id.items()}

argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu',type=int,default=0)
argparser.add_argument('--i_beg',type=int,default=0)
argparser.add_argument('--i_end',type=int,default=15)
args = argparser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

epochs = 10
batch_size = 8
learning_rate = 1e-3
lora_rank = 8
# run_name = f'videomae_base_peft_dc_anx_e{epochs}_bs{batch_size}_lr{learning_rate}_r{lora_rank}'
run_name = f'videomae_base_peft_anx_e{epochs}_bs{batch_size}_lr{learning_rate}_r{lora_rank}'
model_name = 'MCG-NJU/videomae-base'
root_d = pathlib.Path('ROOT_D') / sys.argv[1]
prompts_user_d = root_d / 'prompts_user'
train_val_d = root_d / 'train_val'
run_output_d = root_d / run_name
run_output_d.mkdir(parents=True,exist_ok=True)

train_fs = list(sorted(train_val_d.glob('train_anx_*.jsonl')))
val_fs = list(sorted(train_val_d.glob('val_anx_*.jsonl')))
assert len(train_fs) == len(val_fs) == 15

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name,
                                                         trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name,
                                    trust_remote_code=True)

def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    labels,predictions = (labels > 0),(predictions > 0)
    metrics= {'accuracy':skm.accuracy_score(labels,predictions),
              'balanced_accuracy':skm.balanced_accuracy_score(labels,predictions),
              'f1':skm.f1_score(labels,predictions,average='weighted'),
              'precision':skm.precision_score(labels,predictions,average='weighted'),
              'recall':skm.recall_score(labels,predictions,average='weighted')}
    return metrics

def collate_fn(batch):
    # Permute to (num_frames,num_channels,height,width).
    return {"pixel_values":torch.stack([e["video"].permute(1,0,2,3) for e in batch]), 
            "labels":torch.tensor([e["label"] for e in batch])}

def create_dataset(f, split):
    stride = 4
    fps = 30
    n_frames = config.num_frames
    clip_dur = n_frames*stride/fps
    mean = feature_extractor.image_mean
    std = feature_extractor.image_std
    H,W = feature_extractor.size['shortest_edge'],feature_extractor.size['shortest_edge']
    if split == 'train':
        transform = Compose([ApplyTransformToKey(key="video",
                                                 transform=Compose([UniformTemporalSubsample(n_frames),
                                                                    Lambda(lambda x: x / 255.0),
                                                                    Normalize(mean,std),
                                                                    RandomShortSideScale(256,320),
                                                                    CenterCrop((H,W)),
                                                                    RandomHorizontalFlip(p=0.5)]))])
    else:
        transform = Compose([ApplyTransformToKey(key="video",
                                                 transform=Compose([UniformTemporalSubsample(n_frames),
                                                                    Lambda(lambda x: x / 255.0),
                                                                    Normalize(mean,std),
                                                                    Resize((H,W))]))])
    lines = f.open().readlines()
    dataset = list()
    for line in tqdm(lines,desc=f'create dataset [{f.name}]',leave=False):
        example = json.loads(line.strip())
        for video_f in sorted(prompts_user_d.glob(f'{example["participant_id"]}_q??_360x360_30fps.mp4')):
            dataset.append((str(video_f),{'label':label_to_id[example['label']]}))
    clip_sampler = make_clip_sampler('random',clip_dur)
    dataset = LabeledVideoDataset(dataset,
                                  clip_sampler=clip_sampler,
                                  transform=transform,
                                  decode_audio=False)
    return dataset

def train_and_evaluate(train_f, val_f):
    output_d = run_output_d / train_f.stem.replace('train','').replace('_','')
    if not output_d.is_dir():
        print(f'{output_d} does not exists')
        return
    checkpoint_d = [(int(d.name.replace('checkpoint-','')),d)
                    for d in output_d.glob('checkpoint-*')]
    checkpoint_d.sort()
    checkpoint_d = checkpoint_d[-1][1]
    results_f = output_d / 'results_binary.json'
    dataset_train = create_dataset(train_f,'train')
    dataset_val = create_dataset(val_f,'val')
    dataset = {'train':dataset_train,
               'val':dataset_val}
    model = AutoModelForVideoClassification.from_pretrained(checkpoint_d,
                                                            num_labels=len(label_to_id),
                                                            id2label=id_to_label,
                                                            label2id=label_to_id,
                                                            torch_dtype='auto',
                                                            trust_remote_code=True)
    # peft_config = LoraConfig(r=lora_rank,
    #                          target_modules='all-linear') # ['dense','classifier']
    # model = get_peft_model(model,peft_config)
    # model.print_trainable_parameters()
    model = model.cuda()
    print(f'trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # We need to use labels_names=['labels'] to tell the Trainer to look for 'labels'
    # during evaluation. I found this out by tracing the Trainer code.
    training_args = TrainingArguments(output_dir=str(output_d),
                                      eval_strategy='epoch',
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      num_train_epochs=epochs,
                                      max_steps=(dataset_train.num_videos//batch_size)*epochs,
                                      logging_steps=1,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      report_to='none',
                                      dataloader_num_workers=2,
                                      dataloader_prefetch_factor=2,
                                      run_name=f'{run_name}_{output_d.name}',
                                      label_names=['labels'],
                                      remove_unused_columns=False)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=collate_fn,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset, # dataset['val'],
                      compute_metrics=compute_metrics)
    results = trainer.evaluate(dataset)
    # model.print_trainable_parameters()
    json.dump(results,open(results_f,'w'),indent=2)

pbar = tqdm(list(zip(train_fs,val_fs))[args.i_beg:args.i_end],desc=run_name)
for train_f,val_f in pbar:
    assert train_f.stem.replace('train','val') == val_f.stem
    pbar.set_postfix({'train':train_f.name,'val':val_f.name})
    train_and_evaluate(train_f,val_f)
