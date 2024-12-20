import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW

bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"
os.makedirs(model_dir, exist_ok=True)

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
        entities = item['entities']

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
            'entities': entities,
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

# Load Data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

train_data = load_data("/workspace/data/trainingData.json")
validation_data = load_data("/workspace/data/validationData.json")

intent_labels = list(set([item['intent'] for item in train_data]))
entity_labels = list(set([key for item in train_data for key in item['entities'].keys()]))

intent_to_idx = {label: idx for idx, label in enumerate(intent_labels)}
entity_to_idx = {label: idx for idx, label in enumerate(entity_labels)}

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 64

train_dataset = IntentEntityDataset(train_data, tokenizer, max_length)
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
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels = torch.tensor([intent_to_idx[label] for label in batch['intent']]).to(device)
        entity_labels = torch.zeros((len(batch['entities']), len(entity_labels))).to(device)

        for i, entities in enumerate(batch['entities']):
            for entity, value in entities.items():
                entity_labels[i, entity_to_idx[entity]] = 1

        optimizer.zero_grad()

        intent_logits, entity_logits = model(input_ids, attention_mask)

        loss_intent = criterion_intent(intent_logits, intent_labels)
        loss_entity = criterion_entity(entity_logits, entity_labels)
        loss = loss_intent + loss_entity

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Save Model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"Model training complete and saved to {model_dir}.")
