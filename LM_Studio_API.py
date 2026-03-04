# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:44:45 2025

@author: User
"""
import requests
import json
import pandas as pd
from tqdm import tqdm
import time

# Load your dataset
df = pd.read_csv("10000_test.csv")

# LM Studio OpenAI-compatible chat completions endpoint (default port 1234)
url = "http://localhost:1234/v1/chat/completions"

headers = {"Content-Type": "application/json"}

def classify_batch(texts):
    """Classify a batch of texts at once."""
    prompts = [
        f"You are a resident deterioration classification model. Classify the following notes as '1' for notes related to deterioration or '0' for all else.\nText: {text}\nAnswer only with 0 or 1."
        for text in texts
    ]
    
    results = []
    for prompt in prompts:
        data = {
            "model": "google/gemma-3-27b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()
            message = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if message and message[0] in ["0", "1"]:
                results.append(int(message[0]))
            else:
                results.append(None)
        except Exception as e:
            print(f"Error processing batch item: {e}")
            results.append(None)
        time.sleep(0.05)  # optional small delay between requests
    return results

# Set batch size
batch_size = 16  # adjust depending on your GPU & model
predictions = []

for i in tqdm(range(0, len(df), batch_size), desc="Classifying notes in batches"):
    batch_texts = df["note_text"].iloc[i:i+batch_size].tolist()
    batch_preds = classify_batch(batch_texts)
    predictions.extend(batch_preds)

df["predicted_label"] = predictions

# Save the output
df.to_csv("classified_output.csv", index=False)
print("Classification completed and saved to classified_output.csv")
