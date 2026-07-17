"""
gs

Lesson: 13
==========

Topic: Context vector
Goal: Use Attention weights to combine value vectors and create a context aware representation..

"""

import torch
import math

torch.manual_seed(7)

words = [
    "The",
    "witness",
    "is",
    "pure",
    "awareness"    
]

embedding_size = 4
embeddings = torch.rand((len(words), embedding_size))

query_weights = torch.rand((embedding_size, embedding_size))
key_weights = torch.rand((embedding_size, embedding_size))
value_weights = torch.rand((embedding_size, embedding_size))

queries = embeddings @ query_weights
keys = embeddings @ key_weights
values  = embeddings @ value_weights

raw_scores = queries @ keys.T

scaled_scores = raw_scores /math.sqrt(embedding_size)

attention_weights = torch.softmax(scaled_scores, dim=-1)

context_vectors = attention_weights @ values

print("=" * 69)
print("words")
print("=" * 69)

for index, word in enumerate(words):
    print(f"{index} -> {word}")

print("\nAttention Weights")
print("=" * 69)
print(attention_weights)

print("\nValue Vectors")
print("=" * 69) 
print(values)

print("\nContext Vectors")
print("=" * 69)
print(context_vectors)

print("\nShapes")
print("=" * 69)

print(f"Attention Weights   :   {attention_weights.shape}")
print(f"Value Vectors       :   {value_weights.shape}")
print(f"Context Vectors     :   {context_vectors.shape}")


print("@=" * 35)
print("=@" * 35)
