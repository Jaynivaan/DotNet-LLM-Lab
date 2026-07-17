"""
gs

Lesson: 15
==========

Topic   :   Transformer Block

Goal    :   Assemble the complete transformer block from the lessons learned so far.
        :   Combine attention, residual connections,
=====================================================================================
"""


import math
import torch
import torch.nn as nn

class SimpleSelfAttention(nn.Module):
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
        #output projections can can be added if needed..
        #Real multihead usually combines all heads and
        #passes them through another learned linear layer.


    def forward(self, embeddings: torch.Tensor) ->torch.Tensor:
        queries = self.query_layer(embeddings)
        keys = self.key_layer(embeddings)
        values = self.value_layer(embeddings)

        scores = queries @ keys.transpose(-2, -1)

        scaled_scores = scores/math.sqrt(self.embedding_size)

        #for decoder only models a casual mask can be added here
        #which will prevent token from looking into future tokens..

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1
        )

        context_vectors = attention_weights @ values

        return context_vectors


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_size:int) -> None:
        super().__init__()

        expanded_size = embedding_size * 4

        self.network = nn.Sequential(
            nn.Linear(
                embedding_size,
                expanded_size    
            ),
            nn.ReLU(),
            nn.Linear(
                expanded_size,
                embedding_size
            ),
        )
            #instead on Just ReLU; GELU, SiLU or SwiGLU can be used..
            #the models like Qwen and Llama use gated Feed-forward method.

    def forward(self, embedding_size: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()

        self.attention = SimpleSelfAttention(
            embedding_size = embedding_size )

        self.feed_forward = FeedForwardNetwork(
            embedding_size=embedding_size)

        self.first_layer_norm = nn.LayerNorm(
            embedding_size)

        self.second_layer_norm = nn.LayerNorm(
            embedding_size)

            #layernorm can be also replaced by RMS Norm

            #dropout also can be added optionally after attention and feed forward so that overfitting can be reduced during training

    def forward(self, embeddings: torch.Tensor)-> torch.Tensor:
        attention_output = self.attention(embeddings)

        after_attention = embeddings + attention_output
             
        normalized_attention = self.first_layer_norm(
                 after_attention)

        feed_forward_output = self.feed_forward(
                 normalized_attention)
             ##it s vital to preserve the attention-enriched representation while adding feed-forward transformation.

        after_feed_forward = (
                 normalized_attention + feed_forward_output )
             
        block_output = self.second_layer_norm(
                 after_feed_forward)

             ###most advanced production quality  models will be using prenorm layers instead of post-norm style flow that we are using here,,

        return block_output




torch.manual_seed(7)

words = [
    "The",
    "witness",
    "is",
    "pure",
    "awareness"
]

embedding_size = 4

embeddings = torch.rand(
    (len(words), embedding_size))

transformer_block = TransformerBlock(
    embedding_size = embedding_size)

output = transformer_block(embeddings)

print("=" * 69)
print("words")
print("=" * 69)

for index, word in enumerate(words):
    print(f"{index} -> {word}")

print("\nInput Embeddings")
print("=" * 69)
print(embeddings)

print ("\nTransformer Block Output")
print("=" * 69)
print(output)

print("\nShape")
print("=" * 69)
print(f"input   :   {embeddings.shape}")
print(f"output  :   {output.shape}")
        

