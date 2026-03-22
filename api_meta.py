import argparse
import traceback
from datetime import datetime,timedelta
import time
import pathlib
import json

from tqdm import tqdm
import google.auth
import google.auth.transport.requests
from openai import OpenAI


parser = argparse.ArgumentParser()
parser.add_argument('--version',type=str,help='Version of Llama',choices=['3.1','3.2'])
parser.add_argument('--prompt_id',type=str,help='Prompt dir name')
parser.add_argument('--i_beg',type=int,help='Start index',default=0)
parser.add_argument('--i_end',type=int,help='End index',default=99999)
args = parser.parse_args()

if args.version == '3.1':
    # Get the endpoint, region, project_id and url from Vertex AI docs and console.
    # https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#unary
    # https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.1-405b-instruct-maas?project=som-nero-phi-schul-icu-cba
    endpoint = 'us-central1-aiplatform.googleapis.com'
    region = 'us-central1'
    project_id = '<PROJECT-ID>'
    url = f'https://{endpoint}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi'
    model = 'meta/llama-3.1-405b-instruct-maas'
elif args.version == '3.2':
    # Get the endpoint, region, project_id and url from Vertex AI docs and console.
    # https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.2-90b-vision-instruct-maas?project=som-nero-phi-schul-icu-cba
    endpoint = 'us-central1-aiplatform.googleapis.com'
    region = 'us-central1'
    project_id = '<PROJECT-ID>'
    url = f'https://{endpoint}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi'
    model = 'meta/llama-3.2-90b-vision-instruct-maas'
else:
    assert False

# The public doc for Llama doesn't have info on how to use OpenAI library with it.
# I got this info from the notebook in the above links.
credentials, _ = google.auth.default()
auth_request = google.auth.transport.requests.Request()
credentials.refresh(auth_request)
credentials_refresh_t = datetime.now()
client = OpenAI(base_url=url,api_key=credentials.token)

root_d = pathlib.Path('<ROOT_D>') / args.prompt_id
prompts_system_d = root_d / 'prompts_system'
prompts_user_d = root_d / 'prompts_user'
responses_d = root_d / 'responses'
responses_d.mkdir(exist_ok=True)
scores_gad_f = root_d / 'scores_gad.txt'
scores_phq_f = root_d / 'scores_phq.txt'

prompts_system_fs = list(sorted(prompts_system_d.glob('*.md')))
prompts_user_fs = list(sorted(prompts_user_d.glob('*.md')))
assert len(prompts_system_fs) == len(prompts_user_fs)
args.i_end = min(args.i_end,len(prompts_system_fs))

tokens = list()
pbar = tqdm(list(zip(prompts_system_fs,prompts_user_fs))[args.i_beg:args.i_end],
            desc=f'[{root_d.stem}: {args.i_beg} - {args.i_end}] prompts')
for prompt_system_f,prompt_user_f in pbar:
    assert prompt_system_f.stem == prompt_user_f.stem
    participant_id = prompt_system_f.stem
    response_json_f = responses_d / f'{participant_id}.json'
    if response_json_f.is_file():
        continue
    prompt_system = prompt_system_f.read_text()
    prompt_user = prompt_user_f.read_text()
    prompt_system = f'{prompt_system}\n\nRespond ONLY in JSON format.'
    assert 'GAD-7' in prompt_system
    assert 'PHQ-9' in prompt_system
    while True:
        try:
            if datetime.now() - credentials_refresh_t > timedelta(minutes=30):
                credentials.refresh(auth_request)
                credentials_refresh_t = datetime.now()
                client = OpenAI(base_url=url,api_key=credentials.token)
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role":"system",
                          "content":prompt_system},
                          {"role":"user",
                           "content":prompt_user}]
            )
            response = completion.model_dump()
            json.dump(response,open(response_json_f,'w'),indent=2)

            tokens.append((datetime.fromtimestamp(completion.created),
                           completion.usage.total_tokens))
            one_min_ago = datetime.now() - timedelta(minutes=1)
            tokens_last_min = sum(n
                                  for t,n in tokens
                                  if t >= one_min_ago)
            pbar.set_postfix_str(f'{tokens_last_min} tokens in last 1 min')
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            time.sleep(2)
        else:
            break