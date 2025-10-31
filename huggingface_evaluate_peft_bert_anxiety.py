import sys
import os
import argparse
import pathlib
import json

from tqdm import tqdm
import numpy as np
import sklearn.metrics as skm
from transformers import (AutoTokenizer,AutoModelForSequenceClassification,
                          TrainingArguments,Trainer,logging)
# from peft import LoraConfig,TaskType,get_peft_model
import datasets

logging.set_verbosity_info()

label_to_id = {'minimal':0,'mild':1,'moderate':2,'severe':3}
id_to_label = {v:k for k,v in label_to_id.items()}

argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu',type=int,default=0)
argparser.add_argument('--i_beg',type=int,default=0)
argparser.add_argument('--i_end',type=int,default=15)
args = argparser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

epochs = 50
learning_rate = 1e-3
lora_rank = 1
run_name = f'bert_base_peft_anx_e{epochs}_lr{learning_rate}_r{lora_rank}'
model_name = 'google-bert/bert-base-uncased'
root_d = pathlib.Path('ROOT_D') / sys.argv[1]
train_val_d = root_d / 'train_val'
run_output_d = root_d / run_name
run_output_d.mkdir(parents=True,exist_ok=True)

train_fs = list(sorted(train_val_d.glob('train_anx_*.jsonl')))
val_fs = list(sorted(train_val_d.glob('val_anx_*.jsonl')))
assert len(train_fs) == len(val_fs) == 15

tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(examples):
    # HuggingFace Trainer expects keys 'label' (integer) and all the keys in 
    # the tokenizer output -- 'input_ids','attention_mask','input_type_ids'.
    result = {'label':[label_to_id[l] for l in examples['label']]}
    result.update(tokenizer(examples['text'], padding="max_length", truncation=True))
    return result

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
    dataset_train = datasets.Dataset.from_json(str(train_f),split='train')
    dataset_val = datasets.Dataset.from_json(str(val_f),split='val')
    dataset = datasets.DatasetDict({'train':dataset_train,
                                    'val':dataset_val})
    tokenized_dataset = dataset.map(tokenize,batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_d,
                                                               num_labels=len(label_to_id),
                                                               id2label=id_to_label,
                                                               label2id=label_to_id,
                                                               torch_dtype='auto')
    # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
    #                          r=lora_rank)
    # model = get_peft_model(model,peft_config)
    # model.print_trainable_parameters()
    model = model.cuda()
    training_args = TrainingArguments(output_dir=str(output_d),
                                      eval_strategy='epoch',
                                      learning_rate=learning_rate,
                                      num_train_epochs=epochs,
                                      logging_steps=1,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      report_to='none',
                                      run_name=f'{run_name}_{output_d.name}')
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset, # tokenized_dataset['val'],
                      compute_metrics=compute_metrics)
    results = trainer.evaluate(tokenized_dataset)
    # model.print_trainable_parameters()
    json.dump(results,open(results_f,'w'),indent=2)

pbar = tqdm(list(zip(train_fs,val_fs))[args.i_beg:args.i_end],desc=run_name)
for train_f,val_f in pbar:
    assert train_f.stem.replace('train','val') == val_f.stem
    pbar.set_postfix({'train':train_f.name,'val':val_f.name})
    train_and_evaluate(train_f,val_f)
