"""
gs

Lesson 02
=========

Topic : loading a Pretrained LLM
Goal : Download and load Qwen model.
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("=" * 69)
print("Loading...Tokenizer...")
print("=" * 69)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer Successfully Loaded!!!")


print("=" * 69)
print("Loading Model...")
print("=" * 69)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
model = model.to(device)

print("\nModel Loaded Successfully!!!\n")

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")

print(f"\nParameters : {model.num_parameters():,}")

