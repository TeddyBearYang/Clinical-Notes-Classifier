# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:59:23 2025

@author: User
"""
import time
import asyncio
import httpx
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

async def classify_batch(client,texts,model):
    """Send a batch of texts in a single request."""
    combined_text = DELIMITER.join(texts)
    
    # Limit text length
    if len(combined_text) > MAX_CHARS:
        combined_text = combined_text[:MAX_CHARS]
    
    # Prompt for the LLM
    prompt = (
        "You are a clinical triage classification model.\n"
        "Your task is to determine whether each clinical note indicates a need for hospitalisation check up.\n\n"
        "Classification rules:\n"
        "- Output '1' if the patient requires hospitalisation.\n"
        "- Output '0' if hospitalisation is NOT required.\n"
        "- If the note is ambiguous, incomplete, or lacks clear evidence for hospitalisation, classify as '0'.\n"
        "- It is expected most of the notes should be labelled as hospitalisation is required for a check up.\n\n"
        "Output format:\n"
        "- Output exactly ONE digit ('0' or '1').\n"
        "- Do NOT include explanations, labels, or extra text.\n"
        "The note starts with 'NOTE TEXT BEGIN:' and ends with 'NOTE TEXT END'.\n\n"
        f"{combined_text}"
    )
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    
    try:
        response = await client.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        message = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        # Split output into lines
        results = []
        for line in message.splitlines():
            line = line.strip()
            if line and line[0] in ["0", "1"]:
                results.append(int(line[0]))
            else:
                results.append(str(line))
        # Safety: make sure the batch length matches
        if len(results) != len(texts):
            results = [None] * len(texts)
        return results
    except Exception as e:
        print(f"Error in batch: {e}")
        return [None] * len(texts)

# Process all texts in batches asynchronously
async def classify_all_batches(texts,model):
    # Prepare batch slices
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    results = []
    
    async with httpx.AsyncClient() as client:
        for i in tqdm_asyncio(range(0, len(batches), CONCURRENT_BATCHES), desc="Classifying notes"):
            # Get the next set of concurrent batches
            concurrent = batches[i:i+CONCURRENT_BATCHES]
            tasks = [classify_batch(client,b,model) for b in concurrent]
            batch_results = await asyncio.gather(*tasks)
            # Flatten the batch results into a single list
            for br in batch_results:
                results.extend(br)
    
    return results

# Load the dataset
df = pd.read_csv("SCC_hospitalisations_NM Review.csv")

# LM Studio API endpoint
url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}

BATCH_SIZE = 1  # number of rows per batch
DELIMITER = "\n---\n"  # separator between texts in a batch
CONCURRENT_BATCHES = 1  # number of batches to process in parallel
MAX_CHARS = 3000 # maximum number of characters for each text

# Specify LLM to use
models = ["google/gemma-3-1b",
          "google/gemma-3-12b",
          "google/gemma-3-27b",
          "openai/gpt-oss-20b"]

# Remove all "====" note seperators
df["Note text"] = df["Note text"].str.replace("====", "", regex=False)

# Add note text begin and end strings to each note text to aid model
texts = [f"NOTE TEXT BEGIN:\n{t.strip()}\nNOTE TEXT END" for t in df["Note text"].astype(str).tolist()]

# Run all four models
for model in models:
    # Start timer
    start = time.perf_counter()
    predictions = asyncio.run(classify_all_batches(texts,model))
    
    # Save predictions
    df[model] = predictions
    
    # Print run time
    end = time.perf_counter()
    print(f"Model '{model}' completed in {end - start:.3f} seconds")
    
# Export results
df.to_csv("classified_output.csv", index=False)
print("Classification completed and saved to classified_output.csv")
