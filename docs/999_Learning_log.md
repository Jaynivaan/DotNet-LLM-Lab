##gs



daY1
=====
-LLm
=====

- An  LLm is made of tokenizer, transformer and learned weights.
- hugging face stores models as repositories
- `model.safetensors` contains the learned parameters.
- `config.json` describes the architecture.

###Question
 > why embeddings needed?
 > why models predict one token at a time? 

 ###
 Module 1 === LLM Foundations
 ############################

 ***status : completed>>>
 ***duration: one day.

 ##Objective:

 Build a complete local AI development environment and understand the complete journey from human text to a pretrained language model generating a response.

 ---

 #Lessons completed:

 ##Lesson 01: Environment set up

- installed two versions of python.(3.14 and 3.12)
- created and activated .venv 
- Configured powershell for python development
- Installed GPU enabled pytorch from pytorch.org
- verified cuda support
- confirmed NVIDIA RTX 4060 GPU is being used.

--
#Lesson 02: Loading a Pretrained Model

- Learned about pretrained model
- difference between tokenizer and model
- loaded Qwen2.5-0.5B locally
- Used Hugging face transformers
- Loaded the model on to GPU

A pretrained model is a neural network whose knowledge is stored in learned weights.

---
#Lesson 03: Text Generation

- prompting a local LLM
- Using `model.generate()`
- understanding temperature.
- understanding max_new_tokens
- Decoding generated tokens into text


An LLM generates text one token at a time by predicting the most probable next token..

---
#Lesson 04: Tokenizer Basics

- what is  tokenizer.
- difference between words , tokens and token Ids
- Encoding Text
- Decoding Token Ids
- Vocabulary

Transformers never reads words directly. it only processes tokenIds.

```
Human Text => Tokenizer => TokenIds => Transformer
```
---
#Lesson 05: Embeddings

- Token ids have no semantic meaning by themselves..
- embedding layer converts IDs into vectors.
- Embedding vectors are learned during pretraining .. 
- Hidden size determines vector length.

Token ids are addresses..
Embeddings contain  learned  numerical representations that the transformer can understand.

```
tokenIds	=>	Embeddinglayer	=>	Vectors
```
---
#Lesson 06: Model Configuration

- Model configuration inspection
- Hidden size
- Attention heads
- hidden layers
- context length
- vocabulary size

Every llm has a defined architecture that determines its capacity, memory usage,  and reasoning capacity.

---

==========================
Module ###2
==========================

#lesson: 07 self-attention

- this is vital component of modern llm architecture..
- Query(Q) what am i looking for
- Key(K) what information do i contain
- Value(Value) What information do i give them..
- 

