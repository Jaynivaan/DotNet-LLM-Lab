"""
gs

Lesson: 10
==========

Topic: Query, Key and Value

Goal: understand how transformer creates queries, keys and values from embeddings..
=======================================================================================
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

embedding_size = 4

embeddings = torch.rand((len(words), embedding_size))

query_weights = torch.rand((embedding_size, embedding_size))
key_weights = torch.rand((embedding_size, embedding_size))
value_weights = torch.rand((embedding_size, embedding_size))

queries = torch.matmul(embeddings, query_weights)
keys = torch.matmul(embeddings, key_weights)
values = torch.matmul(embeddings, value_weights)

print ("=" * 69)
print ( "words")
print ("=" * 69)

for index, word in enumerate(words):
    print(f"{index} -> {word}")

print("\nEmbeddings")
print("=" * 69)
print(embeddings)

print ("\nQuery Vectors")
print("=" * 69)
print(queries)

print ("\nKey Vectors")
print("=" * 69)
print(keys)

print("\nValue Vectors")
print("=" * 69)
print(values)

print("\nShapes")
print("=" * 69)
print(f"Embeddings  :   {embeddings.shape}")
print(f"Queries     :   {queries.shape}")
print(f"Keys        :   {keys.shape}")
print(f"Values      :   {values.shape}")

print("*" * 69)
print("*" * 69)





