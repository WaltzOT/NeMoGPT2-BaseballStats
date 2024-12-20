import os
import sys
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

# Bot name as input
bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"

# Paths
train_data_path = "/workspace/data/trainingData.json"
validation_data_path = "/workspace/data/validationData.json"

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
print(f"Training GPT-2 model for bot '{bot_name}'...")
trainer.train()
print("Training complete. Saving model...")

# Save the model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Model saved successfully at {model_dir}.")

# Convert to ONNX format
onnx_path = os.path.join(model_dir, f"{bot_name}.onnx")

def save_to_onnx(model, tokenizer, output_path):
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128))  # Example input
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
        opset_version=14,
        #verbose = True,
    )
    print(f"ONNX model saved at {output_path}")

save_to_onnx(model, tokenizer, onnx_path)
