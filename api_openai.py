import sys
import traceback
from datetime import datetime,timedelta
import pathlib
import json

from tqdm import tqdm
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

choice_to_score = {"Not-at-all":"0",
                   "Several-days":"1",
                   "More-than-half-the-days":"2",
                   "Nearly-every-day":"3"}
response_format = {'gad':GAD_7,
                   'phq':PHQ_9,
                   'both':GAD_7_AND_PHQ_9}

def scores_gad(participant_id, answers):
    return (f'{participant_id} '
            f'{choice_to_score[answers.question_1]},'
            f'{choice_to_score[answers.question_2]},'
            f'{choice_to_score[answers.question_3]},'
            f'{choice_to_score[answers.question_4]},'
            f'{choice_to_score[answers.question_5]},'
            f'{choice_to_score[answers.question_6]},'
            f'{choice_to_score[answers.question_7]}')

def scores_phq(participant_id, answers):
    return (f'{participant_id} '
            f'{choice_to_score[answers.question_1]},'
            f'{choice_to_score[answers.question_2]},'
            f'{choice_to_score[answers.question_3]},'
            f'{choice_to_score[answers.question_4]},'
            f'{choice_to_score[answers.question_5]},'
            f'{choice_to_score[answers.question_6]},'
            f'{choice_to_score[answers.question_7]},'
            f'{choice_to_score[answers.question_8]},'
            f'{choice_to_score[answers.question_9]}')

# logging.basicConfig(level=logging.DEBUG)
client = OpenAI(api_key='<API-KEY>')
root_d = pathlib.Path('<ROOT_D>') / sys.argv[1]
# mode = 'gad'
# mode = 'phq'
mode = 'both'

prompts_system_d = root_d / 'prompts_system'
prompts_user_d = root_d / 'prompts_user'
responses_d = root_d / 'responses_gpt_4o'
responses_d.mkdir(exist_ok=True)
scores_gad_f = root_d / 'scores_gad.txt'
scores_phq_f = root_d / 'scores_phq.txt'

prompts_system_fs = list(sorted(prompts_system_d.glob('*.md')))
prompts_user_fs = list(sorted(prompts_user_d.glob('*.md')))
assert len(prompts_system_fs) == len(prompts_user_fs)

tokens = list()
pbar = tqdm(list(zip(prompts_system_fs,prompts_user_fs)),desc='participant')
for prompt_system_f,prompt_user_f in pbar:
    assert prompt_system_f.stem == prompt_user_f.stem
    participant_id = prompt_system_f.stem
    response_json_f = responses_d / f'{participant_id}.json'
    if response_json_f.is_file():
        continue
    prompt_system = prompt_system_f.read_text()
    prompt_user = prompt_user_f.read_text()
    while True:
        try:
            completion = client.beta.chat.completions.parse(
                model='gpt-4o',
                messages=[{"role":"system",
                          "content":prompt_system},
                          {"role":"user",
                           "content":prompt_user}],
                response_format=response_format[mode]
            )
            answers = completion.choices[0].message.parsed
            if mode == 'gad':
                assert 'GAD-7' in prompt_system
                scores = scores_gad(participant_id,answers)
                open(scores_gad_f,'a').write(scores+'\n')
            elif mode == 'phq':
                assert 'PHQ-9' in prompt_system
                scores = scores_phq(participant_id,answers)
                open(scores_phq_f,'a').write(scores+'\n')
            else:
                assert 'GAD-7' in prompt_system
                assert 'PHQ-9' in prompt_system
                scores = scores_gad(participant_id,answers.GAD7)
                open(scores_gad_f,'a').write(scores+'\n')
                scores = scores_phq(participant_id,answers.PHQ9)
                open(scores_phq_f,'a').write(scores+'\n')
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
        else:
            break