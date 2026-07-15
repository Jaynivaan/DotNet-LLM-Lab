"""
gs

Lesson 01
=========

Topic : Hugging face Tokenizer
Goal: Convert text into tokens and tokenIds
"""



from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("=" * 60)
print("Loading.. Tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "I love how large language models work."

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text)

decoded =  tokenizer.decode(token_ids)

print ("\nOriginal Text: ")
print (text)

print ("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nDecoded Text: ")
print(decoded)