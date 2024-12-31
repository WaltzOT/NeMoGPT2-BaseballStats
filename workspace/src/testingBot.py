import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Bot name as input
bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"
test_data_path = "/workspace/data/testData.json"

# Load Model and Tokenizer
print(f"Loading model for bot '{bot_name}'...")
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Load Test Data
def load_test_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

test_data = load_test_data(test_data_path)

# Generate and extract structured responses
def generate_response(query):
    input_text = f"Query: {query}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).input_ids
    output_ids = model.generate(input_ids=input_ids, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Testing Loop
print("Generating responses for test cases...")
for idx, item in enumerate(test_data):
    query = item["query"]
    print(f"Test Case {idx+1}:")
    print(f"Input: {query}")
    response = generate_response(query)
    
    # Parse the response to extract intent and entities
    if "intent:" in response.lower():
        print(f"Generated Response: {response.strip()}")
    else:
        print(f"Generated Response: {response.strip()}")
    print("-" * 50)

print("Inference complete.")
