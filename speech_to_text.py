import json
import pathlib

from tqdm import tqdm
import numpy as np
import torch
import whisper


data_d = pathlib.Path('/data/depression_anxiety')
audio_fs = list(sorted(data_d.glob('*.mp3')))[700:]
model = whisper.load_model('large-v3',device=torch.device('cuda'))
for audio_f in tqdm(audio_fs):
    transcript_f = audio_f.with_suffix('.json')
    if transcript_f.is_file():
        continue
    transcript = model.transcribe(str(audio_f))
    transcript = json.dumps(transcript,indent=2)
    open(transcript_f,'w').write(transcript)