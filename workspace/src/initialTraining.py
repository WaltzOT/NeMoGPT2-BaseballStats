import os
import json
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Paths
train_data_path = "/workspace/data/trainingData.json"
validation_data_path = "/workspace/data/validationData.json"
model_dir = "/workspace/models/gpt2_nlp_model"

# Load Data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    formatted_data = [{"text": f"Query: {item['query']} | Intent: {item['intent']} | Entities: {json.dumps(item['entities'])}"} for item in data]
    return Dataset.from_list(formatted_data)

train_dataset = load_data(train_data_path)
validation_dataset = load_data(validation_data_path)

# Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    no_cuda=True  # Use CPU
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    data_collator=data_collator,
)

# Train the model
print("Training GPT-2 model for intent classification and entity extraction...")
trainer.train()
print("Training complete. Saving model...")

# Save the model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print("Model saved successfully.")
