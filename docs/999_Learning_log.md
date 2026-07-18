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
# Lesson: 09 Positional Encoding

###Objective:
understand why positional information is added to embeddings
and how it help a transformer understand the order of words.


##What i learned
---

- Transformers process all tokens at the same time, so they donot naturally know the order of words.
- Positional encoding gives each token information about its  relative positioning with in the sentence.
- a position vector is added to each embedding to weave a position aware embedding.
- The same word appearing in different positions will have different final embedding ..
- This address awareness help transformer to distinguish between sentences that contain the same words but different order.

---

#codedecodeing
-----

create random embedding vectors
`size (5, 5)`
- ** 5 rows** one embedding for each of five words.
- ** columns** Each embedding or row hold five numerical features( embedding dimensions)

```python
embeddings = torch.rand(size=(5,5))
```
---

creating the position number for each word / token id
- `end=5` ie , generates positions **0, 1, 2, 3, 4**.
- `unsqueeze(dim=1)` ie, adds a column dimentions so each position becomes its own separate row.
- `float()` ie, convert values to floating point numbers so they can be  added to the embeddings.

```python
positions = torch.arange(5).unsqueeze(1).float()
```

---


Creating a position vector for every word

- `repeat (1, embeddings.shape[1])` ie, `1` -> do not repeat rows; `embeddings.shape[1]` -> Repeat each position across all embedding dimensions(here we configured for dim =5 for each token)
- `0.1` scales the position values to keep them small compared to embedding values.

```python
position_vectors = positions.repeat(1, embeddings.shape[1]) * 0.1
```

---

add the position vector embeddings to finally producing position aware embeddings.
by binding both together.

```python
final_embeddings = embeddings + position_vectors
```

##Data flow
---

```
original embeddings => position vectors => add both => final embeddings

```

=====================================================================================================================================

#Lesson 10 Query, Key and Value

### Objectives
understand how a transformer converts embeddings into **Query(Q)**, **key(K)** and **Value(V)** vectors before calculating attention.


##What i learned..
---

- Every token starts as an embedding
- The transformer  creates  three new vectors from each embedding.
		- Query= What information is this token looking for?
		- Key=What information does this token offer?
		- Value=The actual information that can be materialized from the token.
- Q, K and V are created by multiplying the embeddings with different weight matrices.
- Although they come from the same embedding, they serve diffeerent purposes during attention.

---
###codeDecodeing.
---

defining the size of each embedding vector
`embedding_size=4 ` ie, each word represented by **4 numerical features**

```python
embedding_size = 4
```
---

creating purely random embeddings for each words.
	- `size = (len(words), embedding_size)`
		- `len(words)`	->	one embedding for each word
		- `embedding_size` -> number of dimensions for each embedding

```python
embeddings= torch.rand(size=(len(words), embedding_size))
```
---

creating three different weight matrices

- *** Query Weights*** -> produce query vectors
- *** Key Weights*** -> produces key vectors
- *** Value Weights*** -> produces Value vectors

Each of these matrices learns a different transformation during training.

```python
query_weights = torch.rand( size = ( embedding_size, embedding_size))
key_weights = torch.rand(size = ( embedding_size, embedding_size))
value_weights = torch.rand(size = (embedding_size, embedding_size))
```

---

Transforms the embeddings into ***Query Vectors****
- `torch.matmul()`  ie matrix multiplication 
- each embedding is multiplied by query weight matrix.

```python
queries = torch.matmul( embeddings, query_weights)
```

Transforms the embeddings into ***key Vectors****

```python
keys = torch.matmul( embeddings, key_weights)
```
Transforms the embeddings into ***Value Vectors****

```python
Value = torch.matmul( embeddings, Value_weights)
```

---

##data flow

```
word/tokenid  => embeddings =>query_weights, key_weights, value_weights 
```
=====================================================================================================================================


##Lesson 11: Attention Scores

###Objectives;
understand how  a transformer measures the relationship between  every pair of words using attention Scores.

What i learned :::
---
- Every word has a Query vector and a Key vector
- A query is compared with every key to measure similarity..
- this similarity values are called attention scores
- high attention scores show stronger relationship 
- every word computes attention scores with every other words in the sequence..
---

#code decoding'
---
defining embedding size
```python
embedding_size = 4
```

creating embedding s for  each tokenid based on defined size which is 4.
`size = (len(words), embedding_size)`
`len(words)` means one embedding for each item in array words
`embedding_size` means addeing the defined size ie, 4 here

```
embedding = torch.rand(size=(len(words), embedding_size))
```

---

creating weight matrices used to generate  Query and Key vectors.
- query weight for  transforming embeddings to query vectors.
- key weights for transforming embeddings to key vectors

```python
query_weights = torch.rand(size=(embedidng_size, embedding_size))
key_weights= torch.rand(size=(embedding_size, embedding_size))
```
--
Generating the query vector

- `@` matrix multiplication
- every embedding dimension s are multiplied by query weight matrix

```python
queries = embeddings @ query_weights
```

Generating the key vector

```python
key = embeddings @ key_weights
```

---

calculating attentionscores

- `queries` represent what each word is searching for..
- `keys.T` transposes the key matrix so that  every query can be compared to every key.
- `@` computes similarity between all query-key pairs
- result will be an attention score matrix..

```python
attention_scores = queries @ keys.T
```
---


###
data flow

```
embeddings => query weights, key weights => queries, keys  => comparing queries and keys => AttentionSCORE matrix
```
======================================================================================

