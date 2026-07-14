"""
gs

Lesson: 05
----------

Topic: Token Embeddings
Goal: learn how token ids are converted into  numerical vectors that the transformer can process.

===========================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 69)
print("Loading Tokenizer and Model....")
print("=" * 69)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

text = "Hello Devi!"

print("\nOriginal Text")
print("-" * 69)
print(text)

encoded = tokenizer(text, return_tensors="pt")
input_ids = encoded["input_ids"].to(device)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print("\nTokens and Token IDs")
print("-" * 69)

for token, token_id in zip(tokens, input_ids[0].tolist()):
    print(f"{token:20} -> {token_id}")

embedding_layer = model.get_input_embeddings()

print("\nEmbedding Layer")
print("-" * 69)
print(embedding_layer)

with torch.no_grad():
    embeddings = embedding_layer(input_ids)

print("\nEmbedding Tensor Shape")
print("=" * 69)
print(embeddings.shape)

print("\nMeaning of the Shape")
print("=" * 69)
print(f"Batch size          :   {embeddings.shape[0]}")
print(f"Number of Tokens    :   {embeddings.shape[1]}")
print(f"Values per Token    :   {embeddings.shape[2]}")

first_token_embedding = embeddings[0,0]

print("\nFirst Token")
print("=" * 69)
print(tokens[0])

print("\nFirst 20 Embedding Values")
print("-" * 69)
print(first_token_embedding[:20].float().cpu())

print("\nTotal Values in one Token Embedding")
print("=" * 69)
print(first_token_embedding.numel())

print("\nDevice")
print("-" * 69)
print(embeddings.device)
