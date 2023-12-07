# Task 1. Natural Language Processing. Named entity recognition
In this task, we need to train a named entity recognition (NER) model for the identification of
mountain names inside the texts. For this purpose you need:
* Find / create a dataset with labeled mountains.
* Select the relevant architecture of the model for NER solving.
* Train / finetune the model.
* Prepare demo code / notebook of the inference results.

## Dataset 
Using recommendations, I decided to create a dataset with ChatGPT-3.5. All steps of creating I described in [data_creation.ipynb](https://github.com/Strongich/ds_intership/blob/main/task1/data_creation.ipynb).
## Model selection
For Name Entity Recognition (NER) task for the identification of mountain names inside text I've used fine-tuned BERT, that was trained on standart <b>CoNLL-2003 Named Entity Recognition</b> dataset. You can find the original model [here](https://huggingface.co/dslim/bert-base-NER).
## Fine-tuning 
Before talking about fine-tuning, I want to indicate that from the original model I've deleted 2 last layers because of lack of video memory on my GPU. 

Our pretrained model had other labels from ours. So in order to specify this model for our task I've trained it on a ChatGPT dataset, previously tokenized. I used AdamW optimizer with learning rate=0.00002, 5 epochs, batch size 4 and weight decay 0.01. Results is:
* eval_f1: 0.897
* eval_accuracy: 0.97 \
All weights of model are stored in huggingface repo: [https://huggingface.co/Strongich/bert-english-ner-mountain](https://huggingface.co/Strongich/bert-english-ner-mountain).\
All steps of tokenizing input data and training I've described in [train.py](https://github.com/Strongich/ds_intership/blob/main/task1/train.py).
## Setup
To use trained model, follow the instructions below:
1. First clone the repository. To do this, open a terminal, go to the directory where you want to clone the project and then enter the command:
```bash
git clone https://github.com/Strongich/ds_intership.git
```
2. Go to folder with project and this task and install virtualenv, write the following command and press Enter:
```bash
cd task1
pip install virtualenv
```
3. Next create a new environment, write the following command and press Enter:
```bash
virtualenv name_of_the_new_env
```
### Example:
```bash
virtualenv ner
```
4. Next activate the new environment, write the following command and press Enter:
```bash
name_of_the_new_env\Scripts\activate
```
### Example:
```bash
ner\Scripts\activate
```
5. Write the following command and press Enter:
 ```bash
pip install -r requirements.txt
```
6. You can now open <b>inference_demo.ipynb</b> notebook and use it, OR write it in console and press Enter:
```bash
python
from inference import run_inference
example = "this is inference test via terminal"
run_inference(example)
```
You can change the sentence "this is inference test..." to your own to try it.
