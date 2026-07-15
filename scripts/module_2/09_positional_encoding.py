"""
gs

Lesson: 09
==========

Topic:  Positional Encoding.
Goal:   Demonstrate how embeddings change when position information is added.
===========================================================
"""


import torch

torch.manual_seed(7)

words = [
    "The",
    "witness",
    "is",
    "pure",
    "awareness"    
]

embeddings = torch.rand((5, 5))

positions = torch.arange(5).unsqueeze(1).float()

position_vectors = positions.repeat(1,embeddings.shape[1]) * 0.1

final_embeddings = embeddings + position_vectors


print("=" * 69)
print("Words")
print("=" * 69)

for i, word in enumerate(words):
    print(f"{i} -> {word}")

print("\nOriginal Embeddings")
print("=" * 69)
print( embeddings )

print("\nPosition Vectors")
print("=" * 69)
print(position_vectors)

print("\nFinal Embeddings")
print("=" * 69)
print(final_embeddings)

