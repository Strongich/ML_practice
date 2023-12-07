import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, TrainingArguments, Trainer, DataCollatorForTokenClassification 
import pandas as pd
import numpy as np
import datasets
import torch
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_dataset(path_to_csv):
    '''
    In this section we creating \
    tokenized dataset, assuming that \
    we already have dataset with 2 columns: 
    1. Sentence - basic str
    2. Labels for each word - str with labels for each word, separated by commas
    '''
    data = pd.read_csv(path_to_csv)
    # we will split 80/20
    train_dataset = data.sample(frac=0.8)
    eval_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("EVAL Dataset: {}".format(eval_dataset.shape))
    # map strs to list of strs
    train_dataset.sentence = train_dataset.sentence.apply(lambda x: x.split())
    train_dataset.word_labels = train_dataset.word_labels.apply(lambda x: x.split(','))
    eval_dataset.sentence = eval_dataset.sentence.apply(lambda x: x.split())
    eval_dataset.word_labels = eval_dataset.word_labels.apply(lambda x: x.split(','))
    labels_to_ids = {'B-Mountain': 0, 'I-Mountain': 1, 'O': 2}
    # function to add new column with corresponding value of label
    def map_words_to_numbers(word_list):
        return [labels_to_ids[word] for word in word_list]

    train_dataset["ner_tags"] = train_dataset.word_labels.apply(map_words_to_numbers)
    eval_dataset["ner_tags"] = eval_dataset.word_labels.apply(map_words_to_numbers)
    # convert datasets from pandas to huggingface's dataset
    train_data = datasets.Dataset.from_pandas(train_dataset)
    test_data = datasets.Dataset.from_pandas(eval_dataset)
    return train_data, test_data
def tokenize_and_align_labels(examples,label_all_tokens=True): 
    '''
    Can be only used in .map method
    '''
    # our tokens (in terms of list of words of sentence) is containing in column "sentence"
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token. 
        previous_word_idx = None 
        label_ids = []
        # Special tokens like `` and `<\s>` are originally mapped to None 
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids: 
            if word_idx is None: 
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token                 
                label_ids.append(label[word_idx]) 
            else: 
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                # mask the subword representations after the first subword
                    
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs 

def train():
    labels_to_ids = {'B-Mountain': 0, 'I-Mountain': 1, 'O': 2}
    data_collator = DataCollatorForTokenClassification(tokenizer) 
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    config = AutoConfig.from_pretrained("dslim/bert-base-NER", num_hidden_layers=model.config.num_hidden_layers - 2, num_labels=len(labels_to_ids))
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", config=config, ignore_mismatched_sizes=True)

    model.to(device)
    metric = datasets.load_metric("seqeval") 
    label_list = ['B-Mountain', 'I-Mountain', 'O']

    def compute_metrics(eval_preds): 
        pred_logits, labels = eval_preds 
        
        pred_logits = np.argmax(pred_logits, axis=2) 
        # the logits and the probabilities are in the same order,
        # so we don’t need to apply the softmax
        
        # We remove all the values where the label is -100
        predictions = [ 
            [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(pred_logits, labels) 
        ] 
        
        true_labels = [ 
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
        results = metric.compute(predictions=predictions, references=true_labels)

        return { 
            "precision": results["overall_precision"], 
            "recall": results["overall_recall"], 
            "f1": results["overall_f1"], 
            "accuracy": results["overall_accuracy"], 
    } 
    train_data, test_data = generate_dataset('./data/train.csv')
    tokenized_train = train_data.map(tokenize_and_align_labels, batched=True)
    tokenized_test = test_data.map(tokenize_and_align_labels, batched=True)
    args = TrainingArguments( 
    "finetune_logs",
    logging_dir='finetune_logs',
    evaluation_strategy = "epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4, 
    num_train_epochs=5, 
    weight_decay=0.01, 
    ) 

    trainer = Trainer( 
    model, 
    args, 
    train_dataset=tokenized_train, 
    eval_dataset=tokenized_test, 
    data_collator=data_collator, 
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics 
    ) 
    trainer.train() 
    # both of folders must be precreated 
    model.save_pretrained("ner_model")
    tokenizer.save_pretrained("tokenizer")


if __name__ == "__main__":
    train()


