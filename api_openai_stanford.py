from datetime import datetime,timedelta
import time
import pathlib
import json

from tqdm import tqdm
import requests
from rich import print_json


url = '<STANFORD-API-ENDPOINT>'
headers = {
    'Ocp-Apim-Subscription-Key':'<API-KEY>',
    'Content-Type': 'application/json'
}
root_d = pathlib.Path('<ROOT_D>')
prompts_system_d = root_d / 'prompts_system'
prompts_user_d = root_d / 'prompts_user'
responses_d = root_d / 'responses_gpt_4o'
responses_d.mkdir(exist_ok=True)
prompts_system_fs = list(sorted(prompts_system_d.glob('*.md')))
prompts_user_fs = list(sorted(prompts_user_d.glob('*.md')))

tokens = list()
pbar = tqdm(list(zip(prompts_system_fs,prompts_user_fs)))
for prompt_system_f,prompt_user_f in pbar:
    assert prompt_system_f.stem == prompt_user_f.stem
    participant_id = prompt_system_f.stem
    response_json_f = responses_d / f'{participant_id}.json'
    if response_json_f.is_file():
        continue
    prompt_system = prompt_system_f.read_text()
    prompt_user = prompt_user_f.read_text()
    payload = json.dumps({"model":"gpt-4o",
                          "messages":[{"role":"system",
                                       "content":prompt_system},
                                      {"role":"user",
                                       "content":prompt_user}],
                          "response_format":{"type":"json_object"}})
    while True:
        response = requests.request("POST",url,headers=headers,data=payload)
        if response.status_code == 200:
            response = response.json()
            json.dump(response,open(response_json_f,'w'),indent=2)
            tokens.append((datetime.fromtimestamp(response['created']),
                            response['usage']['total_tokens']))
            one_min_ago = datetime.now() - timedelta(minutes=1)
            tokens_last_min = sum(n
                                  for t,n in tokens
                                  if t >= one_min_ago)
            pbar.set_postfix_str(f'{tokens_last_min} tokens in last 1 min')
            time.sleep(2)
            break
        elif response.status_code == 429:
            pbar.set_postfix_str(f'retry after {response.headers['Retry-After']}')
            time.sleep(int(response.headers['Retry-After']))
