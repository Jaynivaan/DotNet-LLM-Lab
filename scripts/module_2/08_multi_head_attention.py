"""
gs


Lesson: 08
===========

Topic: Multi-Head Attention
Goal: understand why transformers use multiple attentionHeads  instead of just one.
=======
"""

import torch

torch.manual_seed(7)

tokens = ["Mukunda", "loves", "c#" ]

embeddings = torch.rand((3, 4))

print("=" * 69)
print("Tokens")
print("=" * 69)
print(tokens)


print("\nEmbeddings")
print("=" * 69)
print(embeddings)

print("\nSimulating Four Attention Heads")
print("=" * 69)

for head in range(4):

    weights = torch.rand(3)
    weights = torch.softmax(weights, dim=0)

    context = torch.matmul(weights, embeddings)

    print(f"\nHead {head + 1}")
    print ("Attention Weights")

    for token, weight in zip(tokens, weights ):
        print(f"{token:10} -> {weight.item():.4f}")

    print("Context Vector")
    
    print(context)