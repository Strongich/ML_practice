from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

def run_inference(example):
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained('Strongich/bert-english-ner-mountain')
    tokenizer = AutoTokenizer.from_pretrained('Strongich/bert-english-ner-mountain')
    nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
    ner_results = nlp(example)
    print(ner_results)

if __name__ == "__main__":
    run_inference()