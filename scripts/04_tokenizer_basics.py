"""
gs

Lesson: 04
==========

Topic: Tokenizer Basics
Goal: learn how the llm converts human-readable text into tokens and tokenIds before processing.

Key Concepts
------------
= Text
= Tokens
= Token ids
= Encoding
= decoding
= vocabulary
==========================================================================================
"""

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("=" * 69 )
print ("Loading..Tokenizer...")
print ("=" * 69 )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Hello Narayana, Welcome to the world of devi."

print ("\nOriginal Text")
print("--------------")
print(text)

tokens = tokenizer.tokenize(text)

print ("\nTokens")
print("_______")
for i, token in enumerate (tokens):
    print(f"{i+1:2}. {token}")

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nToken IDs")
print("__________")
for  token, token_id in zip(tokens, token_ids):
    print(f"{token:20} -> {token_id}")

decoded = tokenizer.decode(token_ids)

print("\nDecoded")
print("________")
print(decoded)

print ("\nVocabulary Size")
print("_________________")
print(tokenizer.vocab_size)


