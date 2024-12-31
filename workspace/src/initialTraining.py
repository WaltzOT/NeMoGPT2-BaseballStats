import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

# Define Bot Name and Directories
bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"
os.makedirs(model_dir, exist_ok=True)

# Load Intent and Entity Labels
intent_labels_df = pd.read_csv('/workspace/data/intent_label.csv')
intent_labels = intent_labels_df['intent'].tolist()

entity_labels_df = pd.read_csv('/workspace/data/slot_label.csv')
entity_labels = entity_labels_df['entity'].tolist()

# Define Dataset Class
class IntentEntityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        query = item['query']
        intent = item['intent']
        entities = item.get('entities', {})

        # Format entities into a string
        entity_string = ", ".join(f"{key}: {value}" for key, value in entities.items())
        target_text = f"intent: {intent}; entities: {entity_string}"

        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
            'target_text': target_text,
        }

# Load Data
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

training_data = load_data('/workspace/data/trainingData.json')
validation_data = load_data('/workspace/data/validationData.json')

# Tokenizer and Dataset
tokenizer = T5Tokenizer.from_pretrained('t5-small')
max_length = 64

train_dataset = IntentEntityDataset(training_data, tokenizer, max_length)
val_dataset = IntentEntityDataset(validation_data, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model Initialization
model = T5ForConditionalGeneration.from_pretrained('t5-small')
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training and Validation Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {total_train_loss:.4f}")

    # Validation Loop
    model.eval()
    total_val_loss = 0
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            target_texts = batch['target_text']

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = [target_text for target_text in target_texts]

            predictions.extend(decoded_preds)
            actuals.extend(decoded_labels)

            total_val_loss += model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss.item()

    val_acc = accuracy_score(actuals, predictions)
    print(f"Epoch {epoch + 1}/{epochs} | Validation Loss: {total_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

# Save Model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"Model training complete and saved to {model_dir}.")
