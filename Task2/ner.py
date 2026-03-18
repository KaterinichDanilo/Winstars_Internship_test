import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)
from datasets import Dataset

class AnimalNER:
    def __init__(self, device, model_name="distilbert-base-uncased"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.label_list = ["O", "B-ANIMAL"]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self, json_data, output_dir="./ner_results", epochs=3):
        raw_dataset = Dataset.from_list(json_data)
        tokenized_dataset = raw_dataset.map(self._tokenize_and_align_labels, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=10,
            report_to="none"
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )

        trainer.train()

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        predictions = torch.argmax(outputs, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        found_animals = []
        for token, pred in zip(tokens, predictions[0]):
            if self.id2label[pred.item()] == "B-ANIMAL":
                clean_token = token.replace("##", "")
                found_animals.append(clean_token)

        return list(set(found_animals))

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)