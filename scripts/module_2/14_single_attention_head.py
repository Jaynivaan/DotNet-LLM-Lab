"""
gs

Lesson: 14
==========

Topic: Single Attention Head as a reusable module.

Goal: combine query,key, values, rawattentionscores, scaled scores, softmax, and context vectors into one reusable pytorch class..
=======
"""


import torch
import math
import torch.nn as nn

class SingleAttentionHead(nn.Module):
    def __init__(self, embedding_size: int) -> None: 
        super().__init__()

        self.embedding_size = embedding_size

        self.query_layer = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False            
        )

        self.key_layer = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False            
        )

        self.value_layer = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

    def forward(
        self,
        embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        queries = self.query_layer(embeddings)
        keys = self.key_layer(embeddings)
        values = self.value_layer(embeddings)

        scores = queries @ keys.transpose(-2, -1)

        scaled_scores = scores / math.sqrt(self.embedding_size)

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1)

        context_vectors = attention_weights @ values

        return context_vectors, attention_weights

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

attention_head = SingleAttentionHead(
        embedding_size = embedding_size)

context_vectors, attention_weights = attention_head(embeddings)

print("=" * 69)
print("words")
print("=" * 69)

for index, word in enumerate(words):
    print(f"{index} -> {word}")


print("\nEmbeddings")
print("=" * 69)
print(embeddings)

print("\nAttention Weights")
print("=" * 69)
print(attention_weights)

print("\nContext Vectors")
print("=" * 69)
print(context_vectors)

print("\nShapes")
print("=" * 69)
print(f"Embeddings      :       {embeddings.shape}")
print(f"Attention Weight:       {attention_weights.shape}")
print(f"Context Weight  :       {context_vectors.shape}")

print("=+" * 35)
print("+=" * 35)



    