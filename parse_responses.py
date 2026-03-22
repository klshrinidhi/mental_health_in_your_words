import sys
import traceback
import re
import json
import pathlib

from tqdm import tqdm


choice_to_score = {"Unknown":"0",
                   "Not-at-all":"0",
                   "Several-days":"1",
                   "More-than-half-the-days":"2",
                   "Nearly-every-day":"3"}
root_d = pathlib.Path('ROOT_D') / sys.argv[1]
responses_d = root_d / 'responses'
scores_gad_f = root_d / 'scores_gad.txt'
scores_phq_f = root_d / 'scores_phq.txt'


def is_GAD7(q):
    if 'gad' in q.lower():
        return True
    elif re.search(r'general.*anxiety.*disorder',q.lower()):
        return True
    else:
        return False
    
def is_PHQ9(q):
    if 'phq' in q.lower():
        return True
    elif re.search(r'patient.*health.*questionnaire',q.lower()):
        return True
    else:
        return False

response_fs = list(sorted(responses_d.glob('*.json')))
pbar = tqdm(response_fs,desc='responses')
n_removed = 0
all_scores_gad = list()
all_scores_phq = list()
for response_f in pbar:
    response = json.load(open(response_f))
    if 'choices' in response:
        content = response['choices'][0]['message']['content']
    elif 'candidates' in response:
        content = response['candidates'][0]['content']['parts'][0]['text']
    elif 'response' in response:
        content = response['response']
    if isinstance(content,str):
        # Sometimes the response is a markdown code block with a json tag. Find the
        # real beg & end of the json string within.
        try:
            beg,end = content.index('{'),content.rindex('}')+1
            answers = json.loads(content[beg:end])
        except:
            traceback.print_exc()
            print('#'*80)
            print(content)
            print('#'*80)
            print(response_f)
            break
            # response_f.unlink()
            # n_removed += 1
            # pbar.set_postfix({'n_removed':n_removed})
            # continue
    else:
        answers = content
    if len(answers) == 2:
        for qtype,qanda in answers.items():
            scores = list()
            if is_GAD7(qtype):
                assert len(qanda) == 7,response_f
                if isinstance(qanda,dict):
                    for q,a in qanda.items():
                        if 'answer' in a:
                            a = a['answer']
                        if isinstance(a,dict):
                            print(response_f)
                            # response_f.unlink()
                            # n_removed += 1
                            # pbar.set_postfix({'n_removed':n_removed})
                            sys.exit()
                        scores.append(choice_to_score[a])
                elif isinstance(qanda,list):
                    for qa in qanda:
                        if isinstance(qa,dict):
                            for k,v in qa.items():
                                if 'q' in k.lower():
                                    continue
                                scores.append(choice_to_score[v])
                        else:
                            scores.append(choice_to_score[qa])
                all_scores_gad.append(f'{response_f.stem} '+','.join(scores))
            elif is_PHQ9(qtype):
                assert len(qanda) == 9,response_f
                if isinstance(qanda,dict):
                    for q,a in qanda.items():
                        if 'answer' in a:
                            a = a['answer']
                        if isinstance(a,dict):
                            print(response_f)
                            # response_f.unlink()
                            # n_removed += 1
                            # pbar.set_postfix({'n_removed':n_removed})
                            sys.exit()
                        scores.append(choice_to_score[a])
                elif isinstance(qanda,list):
                    for qa in qanda:
                        if isinstance(qa,dict):
                            for k,v in qa.items():
                                if 'q' in k.lower():
                                    continue
                                scores.append(choice_to_score[v])
                        else:
                            scores.append(choice_to_score[qa])
                all_scores_phq.append(f'{response_f.stem} '+','.join(scores))
            else:
                print(content)
                print('#'*80)
                print(response_f)
                # response_f.unlink()
                # n_removed += 1
                # pbar.set_postfix({'n_removed':n_removed})
                sys.exit()
    elif len(answers) == 16:
        scores_gad = list()
        scores_phq = list()
        for q,a in answers.items():
            if is_GAD7(q):
                if 'answer' in a:
                    a = a['answer']
                if isinstance(a,dict):
                    print(response_f)
                    # response_f.unlink()
                    # n_removed += 1
                    # pbar.set_postfix({'n_removed':n_removed})
                    sys.exit()
                scores_gad.append(choice_to_score[a])
            elif is_PHQ9(q):
                if 'answer' in a:
                    a = a['answer']
                if isinstance(a,dict):
                    print(response_f)
                    # response_f.unlink()
                    # n_removed += 1
                    # pbar.set_postfix({'n_removed':n_removed})
                    sys.exit()
                scores_phq.append(choice_to_score[a])
            else:
                print(content)
                print('#'*80)
                print(response_f)
                # response_f.unlink()
                # n_removed += 1
                # pbar.set_postfix({'n_removed':n_removed})
                sys.exit()
        all_scores_gad.append(f'{response_f.stem} '+','.join(scores_gad))
        all_scores_phq.append(f'{response_f.stem} '+','.join(scores_phq))
    else:
        print(content)
        print('#'*80)
        print(response_f)
        # response_f.unlink()
        # n_removed += 1
        # pbar.set_postfix({'n_removed':n_removed})
        break
open(scores_gad_f,'w').write('\n'.join(all_scores_gad))
open(scores_phq_f,'w').write('\n'.join(all_scores_phq))
