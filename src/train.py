import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from preprocess import load_and_preprocess_data
import os
import json
import numpy as np
# Verileri yükle ve ön işle
train_df, val_df, test_df = load_and_preprocess_data('data/processed/train.csv', 'data/processed/val.csv', 'data/processed/test.csv')

train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

# Metrik hesaplama fonksiyonu
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Model ve tokenizer tanımları
model_names = [
    'dbmdz/bert-base-turkish-uncased',
    'distilbert-base-uncased',
    'roberta-base'
]

tokenizers = {
    'dbmdz/bert-base-turkish-uncased': BertTokenizer,
    'distilbert-base-uncased': DistilBertTokenizer,
    'roberta-base': RobertaTokenizer
}

models = {
    'dbmdz/bert-base-turkish-uncased': BertForSequenceClassification,
    'distilbert-base-uncased': DistilBertForSequenceClassification,
    'roberta-base': RobertaForSequenceClassification
}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

class ContiguousTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        super().save_model(output_dir, _internal_call)

results = {}
for model_name in model_names:
    print(f"Training model: {model_name}")

    tokenizer = tokenizers[model_name].from_pretrained(model_name)
    model = models[model_name].from_pretrained(model_name, num_labels=2)

    for param in model.parameters():
        param.data = param.data.contiguous()

    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(['text'])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text'])

    tokenized_train_dataset.set_format('torch')
    tokenized_val_dataset.set_format('torch')
    tokenized_test_dataset.set_format('torch')

    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',
        evaluation_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=f'./logs/{model_name}',
        logging_steps=100,
        save_total_limit=1,
    )

    trainer = ContiguousTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Validation Accuracy for {model_name}: {eval_results['eval_accuracy']}")
    results[model_name] = eval_results

    predictions, labels, _ = trainer.predict(tokenized_test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    test_accuracy = accuracy_score(test_df['label'], predicted_labels)
    print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")

    results[model_name]['test_accuracy'] = test_accuracy

    # Sınıflandırma raporunu ekle
    report = classification_report(test_df['label'], predicted_labels, output_dict=True)
    results[model_name]['classification_report'] = report

    # Sonuçları modelin bulunduğu klasöre kaydet
    os.makedirs(f'results/{model_name}', exist_ok=True)
    with open(f'results/{model_name}/eval_results.json', 'w') as f:
        json.dump(results[model_name], f)

# Tüm sonuçları tek bir dosyada topla
with open('results/results.json', 'w') as f:
    json.dump(results, f)
