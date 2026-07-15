"""
gs

Lesson 03
=========

Topic : Text Generation
Goal : Ask a pretrained LLM a question.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("=" * 69)
print("Loading...Tokenizer...")
print("=" * 69)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("=" * 69)
print("Loading...Model...")
print("=" * 69)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device choosen is : {device}.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = torch.float16 if device == "cuda" else torch.float32
    )

model = model.to(device)

print (f"{MODEL_NAME} running on {device}.")

prompt = "Explain the meaning of life based on wisdom of yogavasista scripture. explain this like to a child"

print("\nPrompt: ")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=900,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nResponse:")
print("\n")
print(response)