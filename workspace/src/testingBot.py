import os
import sys
import json
import torch
from transformers import BertTokenizer, BertModel

# Bot name as input
bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"
test_data_path = "/workspace/data/testData.json"

# Load Model and Tokenizer
print(f"Loading model for bot '{bot_name}'...")

# Load intent and entity labels
with open(f"{model_dir}/config.json", "r") as f:
    model_config = json.load(f)
intent_labels = model_config["intent_labels"]
entity_labels = model_config["entity_labels"]

tokenizer = BertTokenizer.from_pretrained(model_dir)

class IntentEntityClassifier(torch.nn.Module):
    def __init__(self, intent_labels, entity_labels):
        super(IntentEntityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.intent_classifier = torch.nn.Linear(self.bert.config.hidden_size, len(intent_labels))
        self.entity_classifier = torch.nn.Linear(self.bert.config.hidden_size, len(entity_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        intent_logits = self.intent_classifier(pooled_output)
        entity_logits = self.entity_classifier(pooled_output)
        return intent_logits, entity_logits

model = IntentEntityClassifier(intent_labels, entity_labels)
model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin"))
model.eval()

# Load Test Data
def load_test_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

test_data = load_test_data(test_data_path)

# Generate and extract structured responses
def generate_response(query):
    encoding = tokenizer(
        query,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        intent_logits, entity_logits = model(input_ids, attention_mask)

    intent_idx = torch.argmax(intent_logits, dim=1).item()
    intent = intent_labels[intent_idx]

    entity_scores = torch.sigmoid(entity_logits).squeeze(0).tolist()
    detected_entities = [
        entity_labels[i]
        for i, score in enumerate(entity_scores)
        if score > 0.5
    ]

    return intent, detected_entities

# Testing Loop
print("Generating responses for test cases...")
for idx, item in enumerate(test_data):
    query = item["query"]
    print(f"Test Case {idx+1}:")
    print(f"Input: {query}")
    intent, entities = generate_response(query)
    print(f"Predicted Intent: {intent}")
    print(f"Detected Entities: {entities}")
    print("-" * 50)

print("Inference complete.")