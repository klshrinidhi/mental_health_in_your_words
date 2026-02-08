<div align="center">

<h1>Mental Health in Your Words: Data-Efficient Anxiety and Depression Screening with Large Language Models</h1>

<div>
    Shrinidhi K. Lakshmikanth* &emsp;
    Changan Chen* &emsp;
    Amelia Sattler* &emsp;
    Samira Daswani &emsp;
    Daniela Martinez-Bernal &emsp;
    Juze Zhang &emsp;
    Jenny Xu &emsp;
    Zane Durante &emsp;
    Grace Hong &emsp;
    Steven Lin &emsp;
    Kevin Schulman &emsp;
    Arnold Milstein &emsp;
    Nirav R. Shah &emsp;
    Ehsan Adeli &emsp;
</div>

</div>

</br></br>

## Data Privacy Notice

The data collected from participants in this project constitutes Protected Health Information (PHI) as defined under the Health Insurance Portability and Accountability Act (HIPAA). This data is confidential and cannot be shared publicly or distributed without proper authorization and compliance with HIPAA regulations.

## Setup

Install the dependencies in your Python environment (e.g. virtual environment):
```shell
pip install -r requirements.txt
```

## Files

All the code in this repo requires data to be placed in a directory called `<ROOT_D>` in the code. Additionally, different scripts access different parts of the root directory based on their function. The required sub-directories and files should be clear in the file/script.

`speech_to_text.py`: Used to transcribe audio of the patient interviews to text.

`oneshot_*.py`, `fewshot_*.py`, `audio_zeroshot_*.py`: Used to create prompts including the in-context examples based on experiments.

`api_*.py`: Used to interact with the APIs for different models used in the project.

`parse_responses_*.py`: Used to parse the responses obtained from the APIs.

`huggingface_*.py`: Used to train and evaluate baselines.

`train_ml_algos.py`: Train Autogluon on the results of different LLMs.

`vis_new.ipynb`: Jupyter notebook that fixes the parameters and compiles the final results.