import argparse
import time
import pathlib
import json
import enum

from tqdm import tqdm,trange
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel,GenerationConfig,HarmCategory,HarmBlockThreshold


class Answer(enum.Enum):
    Not_at_all = 'Not-at-all'
    Several_days = 'Several-days'
    More_than_half_the_days = 'More-than-half-the-days'
    Nearly_every_day = 'Nearly-every-day'

    def score(self):
        return {'Not-at-all':0,
                'Several-days':1,
                'More-than-half-the-days':2,
                'Nearly-every-day':3}[self.value]

class GAD_7(BaseModel):
    question_1: Answer
    question_2: Answer
    question_3: Answer
    question_4: Answer
    question_5: Answer
    question_6: Answer
    question_7: Answer

class PHQ_9(BaseModel):
    question_1: Answer
    question_2: Answer
    question_3: Answer
    question_4: Answer
    question_5: Answer
    question_6: Answer
    question_7: Answer
    question_8: Answer
    question_9: Answer

class GAD_7_AND_PHQ_9(BaseModel):
    GAD7: GAD_7
    PHQ9: PHQ_9

# The class GenerationConfig below doesn't accept Pydantic schema if it contains
# `$defs` keys and references in them. The example in the Google docs using 
# typing.TypedDict is not working. This class gives an error simply running the
# example on that page: https://ai.google.dev/gemini-api/docs/structured-output?lang=python
# So I am working around by remvoing the `$defs` keys and replacing the references
# with full object schema. The schema can only contain elements listed here:
# https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Schema
response_schema = GAD_7_AND_PHQ_9.model_json_schema()
response_schema['properties']['GAD7'] = response_schema['$defs']['GAD_7']
response_schema['properties']['PHQ9'] = response_schema['$defs']['PHQ_9']
for i in range(1,8):
    response_schema['properties']['GAD7']['properties'][f'question_{i}'] = response_schema['$defs']['Answer']
for i in range(1,10):
    response_schema['properties']['PHQ9']['properties'][f'question_{i}'] = response_schema['$defs']['Answer']
response_schema.pop('$defs')
json.dump(response_schema,open('/mnt/c/Users/shrik/Downloads/mental_health/response_schema.json','w'),indent=2)
generation_config = GenerationConfig(response_mime_type='application/json',
                                     response_schema=response_schema)

safety_settings = {HarmCategory.HARM_CATEGORY_HARASSMENT:HarmBlockThreshold.BLOCK_NONE,
                   HarmCategory.HARM_CATEGORY_HATE_SPEECH:HarmBlockThreshold.BLOCK_NONE,
                   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:HarmBlockThreshold.BLOCK_NONE,
                   HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:HarmBlockThreshold.BLOCK_NONE,
                   HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY:HarmBlockThreshold.BLOCK_NONE}

def scores_gad(participant_id, answers):
    return (f'{participant_id} '
            f'{answers.question_1.score()},'
            f'{answers.question_2.score()},'
            f'{answers.question_3.score()},'
            f'{answers.question_4.score()},'
            f'{answers.question_5.score()},'
            f'{answers.question_6.score()},'
            f'{answers.question_7.score()}')

def scores_phq(participant_id, answers):
    return (f'{participant_id} '
            f'{answers.question_1.score()},'
            f'{answers.question_2.score()},'
            f'{answers.question_3.score()},'
            f'{answers.question_4.score()},'
            f'{answers.question_5.score()},'
            f'{answers.question_6.score()},'
            f'{answers.question_7.score()},'
            f'{answers.question_8.score()},'
            f'{answers.question_9.score()}')

parser = argparse.ArgumentParser()
parser.add_argument('--location',type=str,help='GCP region',default='us-central1')
parser.add_argument('--prompt_id',type=str,help='Prompt dir name',default='prompts_0_21')
parser.add_argument('--n_chunks',type=int,help='Number of chunks',default=10)
parser.add_argument('--i',type=int,help='Index',default=None)
parser.add_argument('--i_beg',type=int,help='Start index',default=0)
parser.add_argument('--i_end',type=int,help='End index',default=99999)
args = parser.parse_args()

model_name = 'gemini-1.5-pro'
# model_name = 'medlm-large-1.5'
root_d = pathlib.Path('<ROOT_D>') / args.prompt_id
prompts_system_d = root_d / 'prompts_system'
prompts_user_d = root_d / 'prompts_user'
responses_d = root_d / 'responses'
responses_d.mkdir(exist_ok=True)
scores_gad_f = root_d / 'scores_gad.txt'
scores_phq_f = root_d / 'scores_phq.txt'

vertexai.init(location=args.location)

prompts_system_fs = list(sorted(prompts_system_d.glob('*.md')))
prompts_user_fs = list(sorted(prompts_user_d.glob('*.md')))
assert len(prompts_system_fs) == len(prompts_user_fs)

if args.i is not None:
    # Start multiple parallel processes to process all the prompts.
    chunk_sz = int(len(prompts_system_fs) / args.n_chunks) + 1
    args.i_beg = chunk_sz * args.i
    args.i_end = chunk_sz * (args.i + 1)
args.i_end = min(args.i_end,len(prompts_system_fs))

pbar = tqdm(list(zip(prompts_system_fs,prompts_user_fs))[args.i_beg:args.i_end],
            desc=f'[{args.location}] [{root_d.stem}: {args.i_beg} - {args.i_end}] prompts')
for prompt_system_f,prompt_user_f in pbar:
    assert prompt_system_f.stem == prompt_user_f.stem
    participant_id = prompt_system_f.stem
    # pbar.set_postfix({'prompt':participant_id})
    response_json_f = responses_d / f'{participant_id}.json'
    if response_json_f.is_file():
        continue
    prompt_system = prompt_system_f.read_text()
    prompt_user = prompt_user_f.read_text()
    assert 'GAD-7' in prompt_system
    assert 'PHQ-9' in prompt_system
    while True:
        try:
            model = GenerativeModel(model_name=model_name,
                                    generation_config=generation_config,
                                    safety_settings=safety_settings,
                                    system_instruction=prompt_system)
            response = model.generate_content(prompt_user)
            answers = json.loads(response.text)
            answers = GAD_7_AND_PHQ_9(**answers)
            scores = scores_gad(participant_id,answers.GAD7)
            open(scores_gad_f,'a').write(scores+'\n')
            scores = scores_phq(participant_id,answers.PHQ9)
            open(scores_phq_f,'a').write(scores+'\n')
            json.dump(response.to_dict(),open(response_json_f,'w'),indent=2)
        except KeyboardInterrupt:
            raise
        except:
            # If we exceed quota, we will get a 429 error. So we wait for 5 seconds
            # traceback.print_exc()
            for _ in trange(5,desc='waiting',leave=False):
                time.sleep(1)
        else:
            break