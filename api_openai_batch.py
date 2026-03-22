import sys
import time
import pathlib
import json

from tqdm import tqdm,trange
from rich import print_json
from pydantic import BaseModel
from openai import OpenAI


class GAD_7(BaseModel):
    question_1: str
    question_2: str
    question_3: str
    question_4: str
    question_5: str
    question_6: str
    question_7: str

class PHQ_9(BaseModel):
    question_1: str
    question_2: str
    question_3: str
    question_4: str
    question_5: str
    question_6: str
    question_7: str
    question_8: str
    question_9: str

class GAD_7_AND_PHQ_9(BaseModel):
    GAD7: GAD_7
    PHQ9: PHQ_9

client = OpenAI(api_key='<API-KEY>')
batch_sz = 750
root_d = pathlib.Path('<ROOT_D>') / sys.argv[1]
prompts_system_d = root_d / 'prompts_system'
prompts_user_d = root_d / 'prompts_user'
prompts_system_fs = list(sorted(prompts_system_d.glob('*.md')))
prompts_user_fs = list(sorted(prompts_user_d.glob('*.md')))
assert len(prompts_system_fs) == len(prompts_user_fs)
prompts_fs = list(zip(prompts_system_fs,prompts_user_fs))

if len(list(root_d.glob(f'{root_d.name}_batch_??.jsonl'))) == 0:
    payloads = list()
    for prompt_system_f,prompt_user_f in tqdm(prompts_fs,desc=f'[{root_d.name}] making payloads'):
        assert prompt_system_f.name == prompt_user_f.name
        participant_id = prompt_system_f.stem
        prompt_system,prompt_user = prompt_system_f.read_text(),prompt_user_f.read_text()
        messages = [{"role":"system",
                    "content":prompt_system},
                    {"role":"user",
                    "content":prompt_user}]
        payload = json.dumps({
            "custom_id":participant_id,
            "method":"POST",
            "url":"/v1/chat/completions",
            "body":{
                "model":"gpt-4o",
                "messages":messages,
                "response_format":{
                    'type':'json_schema',
                    'json_schema':{
                        'name':'GAD7_PHQ9',
                        'schema':GAD_7_AND_PHQ_9.model_json_schema()
                    }
                }
            }
        })
        payloads.append(payload)

    for i,beg in enumerate(trange(0,len(payloads),batch_sz,desc=f'[{root_d.name}] writing batch files')):
        batch = payloads[beg:beg+batch_sz]
        batch_f = root_d / f'{root_d.name}_batch_{i:02}.jsonl'
        open(batch_f,'w').write('\n'.join(batch))

batch_fs = list(enumerate(sorted(root_d.glob(f'{root_d.name}_batch_??.jsonl'))))
pbar = tqdm(batch_fs,desc=f'[{root_d.name}] batches')
for i_batch,batch_f in pbar:
    batch_output_f = root_d / f'{batch_f.stem}_output.jsonl'
    if batch_output_f.is_file():
        continue
    batch_fh = client.files.create(file=open(batch_f,'rb'),purpose='batch')
    batch_created = False
    attempt = 1
    while True:
        if not batch_created:
            batch = client.batches.create(input_file_id=batch_fh.id,
                                        endpoint='/v1/chat/completions',
                                        completion_window='24h')
            batch_created = True
        batch = client.batches.retrieve(batch.id)
        pbar.set_postfix({'status':batch.status,
                          'total':batch.request_counts.total,
                          'completed':batch.request_counts.completed,
                          'failed':batch.request_counts.failed})
        if batch.status == 'completed':
            break
        elif batch.status == 'failed':
            if batch.errors.data[0].code == 'token_limit_exceeded':
                # Sometimes the batch-api doesn't fails all the requests even though
                # there are no pending jobs. We just have to wait and retry.
                for _ in trange(60,desc=f'[attempt {attempt}] waiting for token limit to reset',leave=False):
                    time.sleep(1)
                batch_created = False
                attempt += 1
                continue
            else:
                print_json(batch.model_dump_json(indent=2))
                sys.exit(1)
        for _ in trange(5,desc=f'[attempt {attempt}] waiting for batch to complete',leave=False):
            time.sleep(1)
    response = client.files.content(batch.output_file_id)
    open(batch_output_f,'w').write(response.text)
    if batch.error_file_id:
        batch_error_f = root_d / f'{batch_f.stem}_error.jsonl'
        response = client.files.content(batch.error_file_id)
        open(batch_error_f,'w').write(response.text)
    # Wait for a minute here. Without this, batches started failing. I had to
    # wait for a full day to get the batches to work again in both the API and
    # the web-ui. Likely their system has a bug or delay in updating the 
    # enqueued tokens. If we submit batches too quickly, the system will think
    # we have exceeedd the token limit.
    if i_batch+1 < len(batch_fs):
        for _ in trange(60,desc='waiting to creating next batch',leave=False):
            time.sleep(1)