"""
gs

Lesson: 18
==========

Topic   :   Tiny GPT

Goal    :   Assemble every thing learned last 07 till 17 lessons and craft a tiny GPT model.
"""


import torch
import torch.nn as nn

torch.manual_seed(7)

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embedding = nn.Embedding(
            32,
            embedding_size
        )

        self.decoder = nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(
        self,
        tokens: torch.Tensor
    ):
        positions = torch.arange(
            tokens.size(1),
            device = tokens.device
        )

        embeddings = (
            self.token_embedding(tokens)
            +
            self.position_embedding(positions)
        )

        output = self.decoder(
            embeddings,
            embeddings
        )

        logits = self.output_layer(output)

        return logits




vocabulary = [
    "The",
    "witness",
    "is",
    "pure",
    "awareness"
]

word_to_id = {
    word    :   index
    for index, word in enumerate(vocabulary)
}

tokens = torch.tensor([
    [
        word_to_id["The"],
        word_to_id["witness"],
        word_to_id["is"],
        word_to_id["pure"],
        word_to_id["awareness"]        
    ]    
],
dtype = torch.long,
)

model = TinyGPT(
    vocab_size = len(vocabulary),
    embedding_size=8
)

logits = model(tokens)

predictions = logits.argmax(dim=-1)

print("=" * 69)
print("Input Tokens")
print("=" * 69)
print(tokens)

print("\nVocabulary")
print("=" * 69)
print(vocabulary)

print("\nLogits Shape")
print("=" * 69)
print(logits.shape)

print("\nPredicted Token Ids")
print("=" * 69)
print(predictions)

print("\nPredicted Words")
print("=" * 69)

for token in predictions[0]:
    print(vocabulary[token.item()])