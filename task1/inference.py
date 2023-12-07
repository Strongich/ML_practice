import json
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
config = json.load(open("ner_model/config.json"))
label2id = {'B-Mountain': 0, 'I-Mountain': 1, 'O': 2}
id2label = {0: 'B-Mountain', 1: 'I-Mountain', 2: 'O'}
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("ner_model/config.json","w"))

def run_inference(example):
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained('Strongich/bert-english-ner-mountain')
    tokenizer = AutoTokenizer.from_pretrained('Strongich/bert-english-ner-mountain')
    nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
    ner_results = nlp(example)
    print(ner_results)

if __name__ == "__main__":
    run_inference()