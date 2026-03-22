import json
import pathlib

from tqdm import tqdm

prompt = r"""You are a Clinical Psychologist assessing patients for anxiety and depression symptoms. You will be provided with a set of questions-answer pairs and audio recordings from our verbal interview of a patient.

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
"""

drop_participant_ids ={1,82,96}
root_d = pathlib.Path('<ROOT_D>')
segments_d = root_d / 'segments'
prompts_d = root_d / 'prompts_system'
prompts_d.mkdir(exist_ok=True)

transcript_fs = list(sorted(segments_d.glob('*.json')))
participant_ids = set()
transcripts = dict()
for transcript_f in tqdm(transcript_fs,desc='transcripts'):
    participant_id,question_id = transcript_f.stem.split('_')
    participant_id,question_id = int(participant_id),int(question_id[1:])
    participant_ids.add(participant_id)
    transcript = json.load(open(transcript_f))['text']
    transcripts[(participant_id,question_id)] = transcript
participant_ids.difference_update(drop_participant_ids)

for participant_id in tqdm(participant_ids,desc='participants'):
    prompt_f = prompts_d/f'{participant_id:03}.md'
    open(prompt_f,'w').write(prompt)