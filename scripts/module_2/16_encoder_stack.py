"""
gs

Lesson: 16
==========

Topic:	Encoder Stack
Goal:	Learn how multiple transformer Blocks are stacked
		to build a deep transformer encoder.

"""
import torch
import torch.nn as nn

torch.manual_seed(7)

class TransformerBlock(nn.Module):
	def __init__(self, embedding_size : int):
		super().__init__()

		self.block = nn.Sequential(

			nn.Linear(
				embedding_size,
				embedding_size
			),

			nn.ReLU(),

			nn.Linear(
				embedding_size,
				embedding_size
			),

			nn.LayerNorm(
				embedding_size
			)
		)
	def forward (
		self,
		embeddings: torch.Tensor
	) -> torch.Tensor:


		return self.block(embeddings)

class EncoderStack(nn.Module):
	def __init__(
		self,
		embedding_size : int,
		number_of_layers: int
	):
		super().__init__()

		self.layers = nn.ModuleList(

			[
				TransformerBlock(
					embedding_size
				)

				for _ in range(number_of_layers)
			]
		)

	def forward(
		self,
		embeddings: torch.Tensor
	)-> torch.Tensor:

		for layer_number, layer in enumerate(self.layers):

			embeddings = layer(embeddings)

			print ("=" * 69)
			print(f"After Encoder Layer {layer_number + 1}")
			print("=" * 69)
			print (embeddings)

		return embeddings

words = [
	"The",
	"witness",
	"is",
	"pure",
	"awareness"
]

embedding_size = 4

embeddings = torch.rand(
	(
		len(words),
		embedding_size
	)
)

encoder = EncoderStack(
	embedding_size=embedding_size,
	number_of_layers=3
)

output = encoder(embeddings)

print("=" * 69)
print("Final Encoder Output")
print("=" * 69)
print(output)

print("\nShape")
print("=" * 69)
print(f"Input	:	{embeddings.shape}")
print(f"output	:	{output.shape}")

print ("=+" * 35)
print ("-" * 69)
print("+=" * 35)


