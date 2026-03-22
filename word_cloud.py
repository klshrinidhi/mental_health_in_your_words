import csv
from collections import defaultdict
import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
plt_px = 1/plt.rcParams['figure.dpi']

questionnaire_f = pathlib.Path('<QUESTIONNAIRE_RESPONSES_CSV>')
transcripts_d = pathlib.Path('<TRANSCRIPTS_D>')
transcript_fs = sorted(transcripts_d.glob('*.json'))
H,W = 1080,1920
bg_color = 'white'
random_state = 108
font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
max_font_size = 600

###################################################################################################

participant_anx = dict()
participant_dep = dict()
reader = csv.DictReader(questionnaire_f.open())
for row in reader:
    participant_id = f'{int(row["participant_id"]):03}'
    if int(row['GAD7_score']) <= 4:
        participant_anx[participant_id] = 'minimal'
    else:
        participant_anx[participant_id] = 'notable'
    if int(row['PHQ9_score']) <= 4:
        participant_dep[participant_id] = 'minimal'
    else:
        participant_dep[participant_id] = 'notable'

participant_resp = defaultdict(list)
for transcript_f in transcript_fs:
    participant,question = transcript_f.stem.split('_')
    if question == 'q01':
        continue
    text = json.load(transcript_f.open())['text']
    participant_resp[participant].append(text)
participant_resp = {k: ' '.join(v) 
                    for k, v in participant_resp.items()}

resp_anx_minimal = '\n'.join([resp.lower() for id,resp in participant_resp.items() if participant_anx[id] == 'minimal'])
resp_anx_notable = '\n'.join([resp.lower() for id,resp in participant_resp.items() if participant_anx[id] == 'notable'])
resp_dep_minimal = '\n'.join([resp.lower() for id,resp in participant_resp.items() if participant_dep[id] == 'minimal'])
resp_dep_notable = '\n'.join([resp.lower() for id,resp in participant_resp.items() if participant_dep[id] == 'notable'])

wordcloud_anx_minimal = WordCloud(width=1920,height=1080,background_color='white',random_state=108,font_path=font_path,max_font_size=max_font_size)
wordcloud_anx_notable = WordCloud(width=1920,height=1080,background_color='white',random_state=108,font_path=font_path,max_font_size=max_font_size)
wordcloud_dep_minimal = WordCloud(width=1920,height=1080,background_color='white',random_state=108,font_path=font_path,max_font_size=max_font_size)
wordcloud_dep_notable = WordCloud(width=1920,height=1080,background_color='white',random_state=108,font_path=font_path,max_font_size=max_font_size)

wc_anx_minimal = wordcloud_anx_minimal.generate(resp_anx_minimal)
wc_anx_notable = wordcloud_anx_notable.generate(resp_anx_notable)
wc_dep_minimal = wordcloud_dep_minimal.generate(resp_dep_minimal)
wc_dep_notable = wordcloud_dep_notable.generate(resp_dep_notable)

def valence(word, threshold=0.1):
    if word in valence.filler_words:
        return 'filler'
    pc = valence.sia.polarity_scores(word)
    if pc['compound'] >= threshold:
        return 'positive'
    elif pc['compound'] <= -threshold:
        return 'negative'
    else:
        return 'neutral'
valence.sia = SentimentIntensityAnalyzer()
valence.filler_words = {
    "um","uh","erm","er","hmm","ah","oh","mmm","right","okay","ok",
    "well","literally","basically","actually","like","yeah","huh"
}

def word_color(word, *args, **kwargs):
    valence_word = valence(word)
    if valence_word == 'positive':
        return '#2ca02c'
    elif valence_word == 'negative':
        return '#d62728'
    elif valence_word == 'neutral':
        return '#7f7f7f'
    elif valence_word == 'filler':
        return '#1f77b4'
    assert False

_ = wc_anx_minimal.recolor(color_func=word_color)
_ = wc_anx_notable.recolor(color_func=word_color)
_ = wc_dep_minimal.recolor(color_func=word_color)
_ = wc_dep_notable.recolor(color_func=word_color)

fig, axes = plt.subplots(2,2,figsize=(1920*plt_px,1080*plt_px))
images = [wc_anx_minimal.to_image(),
          wc_anx_notable.to_image(),
          wc_dep_minimal.to_image(),
          wc_dep_notable.to_image()]
titles = ['Anxiety Minimal',
          'Anxiety Notable',
          'Depression Minimal',
          'Depression Notable']
for ax,im,title in zip(axes.flat,images,titles):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis('off')
plt.subplots_adjust(wspace=0.01)
plt.tight_layout()
plt.show()

svg_anx_minimal = wc_anx_minimal.to_svg(embed_font=True)
svg_anx_notable = wc_anx_notable.to_svg(embed_font=True)
svg_dep_minimal = wc_dep_minimal.to_svg(embed_font=True)
svg_dep_notable = wc_dep_notable.to_svg(embed_font=True)

_ = open('wordcloud_anx_minimal.svg','w').write(svg_anx_minimal)
_ = open('wordcloud_anx_notable.svg','w').write(svg_anx_notable)
_ = open('wordcloud_dep_minimal.svg','w').write(svg_dep_minimal)
_ = open('wordcloud_dep_notable.svg','w').write(svg_dep_notable)