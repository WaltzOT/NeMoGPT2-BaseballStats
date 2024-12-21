import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd

bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"
os.makedirs(model_dir, exist_ok=True)

# Load intent and entity labels from CSV files
intent_labels_df = pd.read_csv('/workspace/data/intent_label.csv')
intent_labels = intent_labels_df['intent'].tolist()

entity_labels_df = pd.read_csv('/workspace/data/slot_label.csv')
entity_labels = entity_labels_df['entity'].tolist()

# Create mappings for intents and entities
intent_to_idx = {label: idx for idx, label in enumerate(intent_labels)}
entity_to_idx = {label: idx for idx, label in enumerate(entity_labels)}


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
        entities = item.get('entities', {})  # Default to an empty dictionary if entities are missing

        # Ensure all entities have keys present in the entity_labels
        entity_vector = {label: 0 for label in entity_labels}
        for entity, value in entities.items():
            if entity in entity_vector:
                entity_vector[entity] = 1  # Mark as present

        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent': intent,
            'entity_vector': list(entity_vector.values()),  # Output vector
        }

# Define the Model
class IntentEntityClassifier(nn.Module):
    def __init__(self, intent_labels, entity_labels):
        super(IntentEntityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, len(intent_labels))
        self.entity_classifier = nn.Linear(self.bert.config.hidden_size, len(entity_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        intent_logits = self.intent_classifier(pooled_output)
        entity_logits = self.entity_classifier(pooled_output)
        return intent_logits, entity_logits

# Prepare Data
with open('/workspace/data/trainingData.json') as f:
    training_data = json.load(f)
with open('/workspace/data/validationData.json') as f:
    validation_data = json.load(f)

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 64

train_dataset = IntentEntityDataset(training_data, tokenizer, max_length)
val_dataset = IntentEntityDataset(validation_data, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model Initialization
model = IntentEntityClassifier(intent_labels, entity_labels)
criterion_intent = nn.CrossEntropyLoss()
criterion_entity = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels_tensor = torch.tensor(
            [intent_to_idx[label] for label in batch['intent']]
        ).to(device)

        # Corrected tensor handling for entity labels
        # Ensure the shapes match by stacking entity_vectors and preserving batch size
        # Ensure the shape of entity_labels_tensor matches entity_logits
        entity_labels_tensor = torch.stack(
           [torch.tensor(vector, dtype=torch.float32) for vector in batch['entity_vector']]
        ).to(device)  # Shape: [batch_size, num_entity_labels]

        # Ensure dimensions are consistent
        entity_labels_tensor = entity_labels_tensor.transpose(0, 1)

        optimizer.zero_grad()

        intent_logits, entity_logits = model(input_ids, attention_mask)

        loss_intent = criterion_intent(intent_logits, intent_labels_tensor)
        loss_entity = criterion_entity(entity_logits, entity_labels_tensor)
        loss = loss_intent + loss_entity

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


# Save Model
save_path = f"{model_dir}"
os.makedirs(save_path, exist_ok=True)

# Save the model weights
torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

# Save the configuration file
model_config = {
    "intent_labels": intent_labels,
    "entity_labels": entity_labels,
    "bert_model": "bert-base-uncased"
}
with open(os.path.join(save_path, "config.json"), "w") as f:
    json.dump(model_config, f)

# Save the tokenizer
tokenizer.save_pretrained(save_path)

print(f"Model training complete and saved to {save_path}.")

