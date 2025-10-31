import sys
import traceback
import json
import pathlib

from tqdm import tqdm

choice_to_score = {"Unknown":"0",
                   "Not-at-all":"0",
                   "Several-days":"1",
                   "More-than-half-the-days":"2",
                   "Nearly-every-day":"3"}
root_d = pathlib.Path('ROOT_D') / sys.argv[1]
responses_d = root_d / 'responses_gpt_4o'
responses_d.mkdir(exist_ok=True)
output_fs = list(sorted(root_d.glob(f'{root_d.stem}_batch_*_output.jsonl')))
scores_gad_f = root_d / 'scores_gad.txt'
scores_phq_f = root_d / 'scores_phq.txt'

all_scores_gad = list()
all_scores_phq = list()
for output_f in tqdm(output_fs,desc='batch-output-file'):
    # if 'batch_14_output' not in output_f.stem:
    #     continue
    responses = open(output_f).readlines()
    for response in tqdm(responses,desc='responses',leave=False):
        try:
            response = json.loads(response)
            response_f = responses_d / f"{response['custom_id']}.json"
            choices = response['response']['body']['choices']
            assert len(choices) == 1
            content = choices[0]['message']['content']
            answers = json.loads(content)
            scores_gad = [choice_to_score[answers['GAD7'][f'question_{q}']]
                          for q in range(1,8)]
            all_scores_gad.append(f'{response_f.stem} '+','.join(scores_gad))
            scores_phq = [choice_to_score[answers['PHQ9'][f'question_{q}']]
                          for q in range(1,10)]
            all_scores_phq.append(f'{response_f.stem} '+','.join(scores_phq))
            json.dump(response['response']['body'],response_f.open('w'),indent=2)
        except:
            traceback.print_exc()
            print('participant-id:',response['custom_id'])
            print('batch-output-file:',output_f)
            break
    else:
        continue
    break
open(scores_gad_f,'w').write('\n'.join(all_scores_gad))
open(scores_phq_f,'w').write('\n'.join(all_scores_phq))
