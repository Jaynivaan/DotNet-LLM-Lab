"""
gs

Lesson: 12
==========

Topic: Scaled Attention Weights
Goals: understand how to convert raw attention Scores into stable probabilities using scaling and softmax,
========================================================================================
"""
import torch
import math

torch.manual_seed(7)

words = [
    "the",
    "witness",
     "is", 
     "pure",
     "awareness"    
]

embedding_size = 4

embeddings = torch.rand((len(words), embedding_size))

query_weights = torch.rand((embedding_size, embedding_size))
key_weights = torch.rand((embedding_size, embedding_size))

queries = embeddings @ query_weights
keys = embeddings @ key_weights

raw_scores = queries @ keys.T

scaled_scores = raw_scores / math.sqrt(embedding_size)

attention_weights = torch.softmax(scaled_scores, dim=-1)

print("=" * 69)
print("Raw Attention Scores")
print("=" * 69)
print (raw_scores)

print("\nScaled Attention Scores")
print("=" * 69)
print(scaled_scores)

print("\nAttention Weights")
print("=" * 69)
print(attention_weights)

print("\nRow Sums")
print("=" * 69 )
print(attention_weights.sum(dim=-1))

print("\nShape")
print("=" * 69)
print(attention_weights.shape)

print("*#" * 36)
print("#*" * 36)


