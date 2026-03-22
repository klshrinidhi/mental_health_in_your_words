from collections import defaultdict
import json
import pathlib
import pickle

from tqdm import tqdm,trange
import vertexai
from vertexai.language_models import TextEmbeddingInput,TextEmbeddingModel


root_d = pathlib.Path('<ROOT_D>')
segments_d = root_d / 'segments'
vertexai.init(location='us-central1')
model = TextEmbeddingModel.from_pretrained('text-embedding-005')
task = 'CLASSIFICATION'
# Default is emb dim is 3072. Following are additional emb dims we need for exps.
extra_emb_dims = [768,512,256]

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
        response = model.get_embeddings([TextEmbeddingInput(transcript,task)],
                                        output_dimensionality=emb_dim)
        embeddings[participant_id][(question_id,emb_dim)] = response[0].values
embeddings_f = root_d / 'embeddings_google_per_answer.pkl'
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
        response = model.get_embeddings([TextEmbeddingInput(all_answers,task)],
                                        output_dimensionality=emb_dim)
        embeddings[participant_id][emb_dim] = response[0].values
embeddings_f = root_d / 'embeddings_google_per_participant.pkl'
pickle.dump(dict(embeddings),open(embeddings_f,'wb'))
