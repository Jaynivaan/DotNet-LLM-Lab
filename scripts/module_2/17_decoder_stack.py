"""
gs

Lesson: 17
==========

Topic   :    Decoder stack

Goal    :   Understand how GPT predicts the next token with out  looking into the future.

"""

from typing import Sequence

import torch
from torch.fx.experimental.dynamic_spec import SeqSpec
import torch.nn as nn

torch.manual_seed(7)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

    def forward(
        self,
        embeddings: torch.Tensor) -> torch.Tensor:



        queries = self.query(embeddings)
        keys = self.key(embeddings)
        values = self.value(embeddings)

        scores = queries @ keys.transpose(-2, -1)

        scores = scores/ (embeddings.shape[-1] ** 0.5)

        sequence_length =  scores.shape[0]

        mask = torch.triu(

              
            torch.ones(
                sequence_length,
                sequence_length
            ),

            diagonal=1

        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        context = attention @ values
            
        return attention, context

words = [
    "The",
    "witness",
    "is",
    "pure",
    "awareness"
]

embedding_size = 4

embeddings = torch.rand((len(words), embedding_size))


decoder = DecoderBlock(embedding_size)

attention, context = decoder (embeddings)

print ("=" * 69)
print("words")
print("=" * 69)

for index, word in enumerate(words):
    print(f"{index} -> {word}")

print("\n Attention Matrix")
print("=" * 69)
print (attention)

print("\n Context Vector")
print("=" * 69)
print(context)


print("\nShape")
print("=" * 69)
print(f"Attention   :   {attention.shape}")
print(f"Context     :   {context.shape}")
