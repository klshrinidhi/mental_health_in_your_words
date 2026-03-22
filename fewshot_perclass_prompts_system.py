import random
from collections import defaultdict
import csv
import json
import pathlib

from tqdm import tqdm

prompt = r"""You are a Clinical Psychologist assessing patients for anxiety and depression symptoms. You will be provided with a set of question-answer pairs from our verbal interview of a patient.

Based on the patient interview, complete the Generalized-Anxiety-Disorder-7 (GAD-7) questionnaire and the Patient-Health-Questionnaire-9 (PHQ-9) questionnaire on behalf of the patient. For each question, answer with one of "Not-at-all", "Several-days", "More-than-half-the-days", or "Nearly-every-day".

# GAD-7 Questionnaire
Over the last two weeks, how often have you been bothered by the following problems?
1. Feeling nervous, anxious, or on edge.
2. Not being able to stop or control worrying.
3. Worrying too much about different things.
4. Trouble relaxing.
5. Being so restless that it is hard to sit still.
6. Becoming easily annoyed or irritable.
7. Feeling afraid as if something awful might happen.

# PHQ-9 Questionnaire
Over the last two weeks, how often have you been bothered by the following problems?
1. Little interest or pleasure in doing things.
2. Feeling down, depressed, or hopeless.
3. Trouble falling or staying asleep, or sleeping too much.
4. Feeling tired or having little energy.
5. Poor appetite or overeating.
6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down.
7. Trouble concentrating on things, such as reading the newspaper or watching television.
8. Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual.
9. Thoughts that you would be better off dead or of hurting yourself in some way.

# Example Patient Interview And Questionnaires
Following are example patient interviews and the corresponding completed GAD-7 and PHQ-9 questionnaires.

{interviews}
"""

anx_fullscore_to_class = dict()
for s in range(22):
    if s <= 4:
        anx_fullscore_to_class[s] = 'minimal'
    elif s <= 9:
        anx_fullscore_to_class[s] = 'mild'
    elif s <= 14:
        anx_fullscore_to_class[s] = 'moderate'
    elif s <= 21:
        anx_fullscore_to_class[s] = 'severe'
dep_fullscore_to_class = dict()
for s in range(27):
    if s <= 4:
        dep_fullscore_to_class[s] = 'minimal'
    elif s <= 9:
        dep_fullscore_to_class[s] = 'mild'
    elif s <= 14:
        dep_fullscore_to_class[s] = 'moderate'
    elif s <= 19:
        dep_fullscore_to_class[s] = 'moderately-severe'
    elif s <= 27:
        dep_fullscore_to_class[s] = 'severe'
score_to_choice = {"0":"Not-at-all",
                   "1":"Several-days",
                   "2":"More-than-half-the-days",
                   "3":"Nearly-every-day"}
questions_interview = [
    'Can you please state your name, the date, and the time?',
    'What is the purpose of your visit?',
    'How are you feeling today?',
    'Please describe how you’ve been feeling emotionally during the last week.',
    'Please describe how you’ve been feeling physically during the last week.',
    'Tell me about your experiences as well as any comments other people made about your emotions.',
    'Please describe how your emotional and physical feelings have affected your general ability to function in the past week. Consider such things as your ability to work, manage your home, get along with others, and participate in leisure activities.',
    'Can you tell us about some recent good news you had and how did that make you feel?',
    'What are some things that usually put you in a good mood?',
    'How often do you feel this way (in a good mood) lately?',
    'Tell me about the last time you felt really happy.'
]
questions_questionnaire_gad = [
    '1. Feeling nervous, anxious, or on edge.',
    '2. Not being able to stop or control worrying.',
    '3. Worrying too much about different things.',
    '4. Trouble relaxing.',
    '5. Being so restless that it is hard to sit still.',
    '6. Becoming easily annoyed or irritable.',
    '7. Feeling afraid as if something awful might happen.'
]
questions_questionnaire_phq = [
    '1. Little interest or pleasure in doing things.',
    '2. Feeling down, depressed, or hopeless.',
    '3. Trouble falling or staying asleep, or sleeping too much.',
    '4. Feeling tired or having little energy.',
    '5. Poor appetite or overeating.',
    '6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down.',
    '7. Trouble concentrating on things, such as reading the newspaper or watching television.',
    '8. Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual.',
    '9. Thoughts that you would be better off dead or of hurting yourself in some way.'
]

num_samples = 200
few_shot = 9
drop_participant_ids ={1,82,96}
root_d = pathlib.Path('<ROOT_D>')
questionnaire_f = root_d / 'questionnaire_data_clean_segmented.csv'
segments_d = root_d / 'segments'
prompts_d = root_d / 'prompts_system'
prompts_d.mkdir(exist_ok=True)

transcript_fs = list(sorted(segments_d.glob('*.json')))
participant_ids = set()
transcripts = dict()
for transcript_f in tqdm(transcript_fs,desc='reading transcripts'):
    participant_id,question_id = transcript_f.stem.split('_')
    participant_id,question_id = int(participant_id),int(question_id[1:])
    participant_ids.add(participant_id)
    transcript = json.load(open(transcript_f))['text']
    transcripts[(participant_id,question_id)] = transcript
participant_ids.difference_update(drop_participant_ids)

print(f'reading questionnaires from {questionnaire_f}')
anx_class_to_participant_ids = defaultdict(list)
dep_class_to_participant_ids = defaultdict(list)
questionnaires = dict()
reader = csv.DictReader(open(questionnaire_f))
for row in reader:
    participant_id = int(row['participant_id'])
    if participant_id in drop_participant_ids:
        continue
    anx_class_to_participant_ids[anx_fullscore_to_class[int(row['GAD7_score'])]].append(participant_id)
    dep_class_to_participant_ids[dep_fullscore_to_class[int(row['PHQ9_score'])]].append(participant_id)
    for q in range(1,8):
        questionnaires[(participant_id,f'GAD7_Q{q}')] = score_to_choice[row[f'GAD7_Q{q}']]
    for q in range(1,10):
        questionnaires[(participant_id,f'PHQ9_Q{q}')] = score_to_choice[row[f'PHQ9_Q{q}']]

print('sampling participants')
random.seed(42)
train_participant_ids_samples = list()
for _ in range(num_samples):
    sample = list()
    for v in anx_class_to_participant_ids.values():
        if len(v) > 1:
            sample.extend(random.sample(v,few_shot))
    train_participant_ids_samples.append(sample)

print('writing prompts')
for val_participant_id in tqdm(participant_ids,desc='participant'):
    assert val_participant_id not in drop_participant_ids
    for train_participant_ids in train_participant_ids_samples:
        interviews = list()
        for i_train_participant_id,train_participant_id in enumerate(sorted(train_participant_ids),1):
            if val_participant_id == train_participant_id:
                break
            assert train_participant_id not in drop_participant_ids

            interview = [f'## Patient {i_train_participant_id} Interview']
            for question_id,question in enumerate(questions_interview,1):
                if not (train_participant_id,question_id) in transcripts:
                    # print(f'Participant {train_participant_id} does not have '
                    #       f'question {question_id}')
                    continue
                question = f'Q: {question}'
                answer = f'A: {transcripts[(train_participant_id,question_id)]}'
                interview.append(question)
                interview.append(answer)
                if question_id < len(questions_interview):
                    interview.append('')
            interviews.append('\n'.join(interview))

            questionnaire_gad = [f'## Patient {i_train_participant_id} Completed GAD-7']
            for question_id,question in enumerate(questions_questionnaire_gad,1):
                answer = f'A: {questionnaires[(train_participant_id,f'GAD7_Q{question_id}')]}'
                questionnaire_gad.append(question)
                questionnaire_gad.append(answer)
            interviews.append('\n'.join(questionnaire_gad))

            questionnaire_phq = [f'## Patient {i_train_participant_id} Completed PHQ-9']
            for question_id,question in enumerate(questions_questionnaire_phq,1):
                answer = f'A: {questionnaires[(train_participant_id,f'PHQ9_Q{question_id}')]}'
                questionnaire_phq.append(question)
                questionnaire_phq.append(answer)
            interviews.append('\n'.join(questionnaire_phq))
        else:
            interviews = '\n\n'.join(interviews)
            prompt_filled = prompt.format(interviews=interviews)
            train_participant_ids = '_'.join(f'{i:03}' for i in train_participant_ids)
            prompt_f = prompts_d / f'val_{val_participant_id:03}_' f'train_{train_participant_ids}.md'
            open(prompt_f,'w').write(prompt_filled)