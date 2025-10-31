import shutil
import json
import pathlib

from tqdm import tqdm

prompt = r"""Following are questions and audio recordings of the patient interview.

{interview}

"""
questions = [
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
root_d = pathlib.Path('<ROOT_D>')
segments_d = root_d / 'segments'
prompts_d = root_d / 'prompts_user'
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

for participant_id in tqdm(participant_ids,desc='participants'):
    audio_d = prompts_d / f'{participant_id:03}'
    audio_d.mkdir(exist_ok=True)
    interview = list()
    for question_id,question in enumerate(questions,1):
        # Skip the first 2 questions.
        if question_id < 3:
            continue
        if not (participant_id,question_id) in transcripts:
            # print(f'Participant {participant_id} does not have question '
            #       f'{question_id}')
            continue
        question = f'Q: {question}'
        interview.append(question)
        src_audio_f = segments_d / f'{participant_id:03}_q{question_id:02}.mp3'
        dst_audio_f = audio_d / src_audio_f.name
        shutil.copyfile(src_audio_f,dst_audio_f)
    interview = '\n'.join(interview)
    
    prompt_filled = prompt.format(interview=interview)
    prompt_f = prompts_d / f'{participant_id:03}.md'
    open(prompt_f,'w').write(prompt_filled)
