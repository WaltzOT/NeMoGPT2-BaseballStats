import os
import sys
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset

# Bot name as input
bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
model_dir = f"/workspace/models/{bot_name}"
test_data_path = "/workspace/data/testData.json"

# Load Model and Tokenizer
print(f"Loading model for bot '{bot_name}'...")
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Load Test Data
def load_test_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

test_data = load_test_data(test_data_path)

# Generate and extract structured responses
def generate_response(query):
    input_text = f"Query: {query} | Intent:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids=input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Testing Loop
print("Generating responses for test cases...")
for idx, item in enumerate(test_data):
    query = item["query"]
    print(f"Test Case {idx+1}:")
    print(f"Input: {query}")
    response = generate_response(query)
    
    if "Intent:" in response:
        intent_part = response.split("Intent:")[1].strip()
        print(f"Generated Response: {intent_part}")
    else:
        print(f"Generated Response: {response}")
    print("-" * 50)

print("Inference complete.")
