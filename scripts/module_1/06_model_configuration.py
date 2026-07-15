"""
gs

Lesson: 06
----------

Topic:  Understanding Model Configuration
Goal:   learn how to inspect pretrained model and discover its architecture..
            ie, to find model configuration, hidden size, number of layers, AttentionHeads, VocabularySize and contextLength.
=============================================================================================================================
"""

from transformers import AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("=" * 69)
print("Loading Model configuration...")
print("=" * 69)

config = AutoConfig.from_pretrained( MODEL_NAME )

print(f"\nModel Name            :     {MODEL_NAME}")
print(f"Vocabulary Size         :     {config.vocab_size}")
print(f"Hidden Size             :     {config.hidden_size}")
print(f"Attention Heads         :     {config.num_attention_heads}")
print(f"Hidden Layers           :     {config.num_hidden_layers}")
print(f"Maximum Positions       :     {config.max_position_embeddings}")

