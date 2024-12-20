import os
import sys
import json
import numpy as np
import torch
import onnxruntime as ort
from transformers import GPT2Tokenizer

# Bot name as input
bot_name = sys.argv[1] if len(sys.argv) > 1 else "default_bot"
onnx_model_path = f"/workspace/models/{bot_name}/{bot_name}.onnx"
test_data_path = "/workspace/data/testData.json"

# Load Test Data
def load_test_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

test_data = load_test_data(test_data_path)

# Load ONNX Model and Tokenizer
print(f"Loading ONNX model for bot '{bot_name}'...")
if not os.path.exists(onnx_model_path):
    print(f"Error: ONNX model for bot '{bot_name}' not found at {onnx_model_path}")
    sys.exit(1)

tokenizer = GPT2Tokenizer.from_pretrained(f"/workspace/models/{bot_name}")
ort_session = ort.InferenceSession(onnx_model_path)

# Generate response using ONNX
import numpy as np

def generate_response(query):
    input_text = f"Query: {query} | Intent:"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()  # Convert to numpy for ONNX

    # Run inference
    outputs = ort_session.run(None, {"input_ids": input_ids})
    logits = outputs[0]  # Assuming this is the logits from the model

    # Debug: Print logits shape and values
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits: {logits[:5]}")  # Print a few logits for inspection

    # Convert logits to token IDs using argmax
    token_ids = np.argmax(logits, axis=-1)  # Get the token IDs with the highest probability

    # Flatten the token IDs if necessary
    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.flatten().tolist()

    # Decode the token IDs
    try:
        response = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as e:
        print(f"Error decoding token_ids: {token_ids}")
        raise e

    return response



# Testing Loop
print("Generating responses for test cases...")
for idx, item in enumerate(test_data):
    query = item["query"]
    print(f"Test Case {idx+1}:")
    print(f"Input: {query}")
    response = generate_response(query)
    print(f"Generated Response: {response}")
    print("-" * 50)

print("Inference complete.")
