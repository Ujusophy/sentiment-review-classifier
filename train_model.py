from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# Load and tokenize
df = pd.read_csv("sentiment_dataset.csv")
dataset = Dataset.from_pandas(df)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def preprocess(example):
    return tokenizer(
        example['text'],
        padding="max_length",       # pad all sequences to max_length
        truncation=True,
        max_length=512              # explicitly set max_length for safety
    )

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.train_test_split(test_size=0.2)

# Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Train
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,,
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
)

trainer.train()

# Save model
model.save_pretrained("model/sentiment_model")
tokenizer.save_pretrained("model/sentiment_model")
