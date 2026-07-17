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

objective:
Understand how  a token decides which other tokens are most important before making a prediction.

##what i learned:
*** self attention is core mechanism of a transformer..
*** Every token compares itself with every other token .
*** Each token creates a  "query" , while every token also  creates a "key" and "value"
*** similarity between the Query and keys produces ""attention scores"".
*** softmax  converts these derived attention scores into """ attention weights"""
*** attention weights are then used to combine the value vectors into a new """context vector"""
*** context vector is a richer representation of token because it now contains information from other relevant tokens.

###code decodeing
-----------------
embedding "sat" asking which tokens are important to me
```python
query = embeddings[2]
```
---

comparing query with every key to calculate relevance.

```python
scores= matmul(keys,query)
```
---

Converts relevance scores into probabilities that sum to **1**

```python
attention_weights = torch.softmax(scores, dim=0)
```
---

creating a new representation called context vector by combining all value vectors according to their importance.

```python
context_vector = torch.matmul(attention_weights, values)
```

```data flow
	embeddings => query=> compare with keys => attentionScores => softmax => attention weights => weighted sum of values => context vector 
```

===========================================================================
#lesson:08	Multi-head Attention

#0bjective=
understand why transformers use multiple attention heads instead of a single attention mechanism.

what i learned
--------------

+ A single attention head captures only one type of relation ship between tokens.
+ Multiple attention heads allow the models to learn different relation ships at the same time.
+ each head calculates its own attention weights and creates its own context vector.
+ The outputs from all heads are then later time combined to form a much richer understanding of the sentence.
+ This helps the model capture syntax, meaning, context, and long-range dependencies simultaneously .


#Code decodeing
---------------

Creating embedding vectors from tokens
```python
embeddings = torch.rand((len(words),embedding_sizee))
## ie(rows,coloumns)\
```
---

Simulates the attentionweights calculated by one attention head.

```python
weights = torch.rand(3)
weights = torch.softmax(weights, dim=0)
```
---

creating context vector for that attention head by combining  the embeddings according to  their importance.

```python
context = torch.matmul(weights, embeddings)
```
 ----
 simulate four  independent attention heads, each producing its own view of the sentence.

 ```python
 for head in range (4):
 ```

 -----
Data flow 

```
Embeddings => Head1 ->context vector 1 =>Head2 -> context vector 2 => Head3 -> context vector 3 => Head4 -> context vector 4 => combined richer context representation .

```

==============================================================================================================================
