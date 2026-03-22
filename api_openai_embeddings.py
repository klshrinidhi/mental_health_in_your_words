from collections import defaultdict
import json
import pathlib
import pickle

from tqdm import tqdm
from openai import OpenAI


client = OpenAI(api_key='<API-KEY>')
root_d = pathlib.Path('<ROOT_D>')
segments_d = root_d / 'segments'
embeddings_d = root_d / 'embeddings_openai'
embeddings_d.mkdir(exist_ok=True)
# Default is emb dim is 3072. Following are additional emb dims we need for exps.
extra_emb_dims = [3072,2048,1536,1024,768,512,256] 

transcript_fs = list(sorted(segments_d.glob('*.json')))
participant_ids,question_ids = set(),set()
transcripts = dict()
for transcript_f in tqdm(transcript_fs,desc='reading transcripts'):
    participant_id,question_id = transcript_f.stem.split('_')
    participant_id,question_id = int(participant_id),int(question_id[1:])
    participant_ids.add(participant_id)
    question_ids.add(question_id)
    transcript = json.load(open(transcript_f))['text']
    transcripts[(participant_id,question_id)] = transcript

embeddings = defaultdict(dict)
for (participant_id,question_id),transcript in tqdm(list(transcripts.items()),desc='answer embeddings'): 
    for emb_dim in tqdm(extra_emb_dims,desc='embedding dims',leave=False):
        if emb_dim == 3072:
            response = client.embeddings.create(input=transcript,
                                                model='text-embedding-3-large')
        else:
            response = client.embeddings.create(input=transcript,
                                                model='text-embedding-3-large',
                                                dimensions=emb_dim)
        response_f = embeddings_d / f'{participant_id:03}_q{question_id:02}_d{emb_dim:04}.json'
        json.dump(response.model_dump(),open(response_f,'w'),indent=2)
        embeddings[participant_id][(question_id,emb_dim)] = response.data[0].embedding
embeddings_f = root_d / 'embeddings_openai_per_answer.pkl'
pickle.dump(dict(embeddings),open(embeddings_f,'wb'))

embeddings = defaultdict(dict)
for participant_id in tqdm(list(sorted(participant_ids)),desc='participant embeddings'):
    all_answers = list()
    for question_id in sorted(question_ids):
        if (participant_id,question_id) not in transcripts:
            continue
        all_answers.append(transcripts[(participant_id,question_id)])
    all_answers = '\n\n'.join(all_answers)
    for emb_dim in tqdm(extra_emb_dims,desc='embedding dims',leave=False):
        if emb_dim == 3072:
            response = client.embeddings.create(input=all_answers,
                                                model='text-embedding-3-large')
        else:
            response = client.embeddings.create(input=all_answers,
                                                model='text-embedding-3-large',
                                                dimensions=emb_dim)
        response_f = embeddings_d / f'{participant_id:03}_d{emb_dim:04}.json'
        json.dump(response.model_dump(),open(response_f,'w'),indent=2)
        embeddings[participant_id][emb_dim] = response.data[0].embedding
embeddings_f = root_d / 'embeddings_openai_per_participant.pkl'
pickle.dump(dict(embeddings),open(embeddings_f,'wb'))
