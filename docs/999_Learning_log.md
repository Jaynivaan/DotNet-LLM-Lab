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

Lesson 12:	Scaled Attention weights

### Objective:
understand how raw attentionscores are converted  into stable posibilities using ***scaling** and **softmax**.

##what i learned:

- Raw attention can become very large as the embedding size increases.
- Large scores makes softmax produce extremely sharp probabilities, making training unstable.
- the act of scaling divides the score by sqrt (embedding_size) inorder to retain them with a reasonable range.
- Softmax t hen convert this scaled scores into probabilities between **0** and **1**.
- Each row of attention weight matrix always sums to **1**, meaning each word distributes 100% of attention scores across all words.
----

##code decodeing
--
Defines the size of embedding vector.
each word token id represented by n number of features.. 
here n is 4 

```python
embedding_size = 4
```

creating random embeddings as previous codes..
```python 
embeddings = torch.rand((size=(len(words), embedding_size))
```

creating queries and keys as in previous code.

```python
queries = embeddings @ query_weights
keys = embeddings @ key_weights
```

creating attention scores by computing the similarity

```python
raw_scores = queries @ keys.T
```

scaling the scores 
so `math.sqrt(embedding_size)`
computes the sqrt of embedding size..
dividing by this value keep the scores from becoming too large, improving training stability.

```python
scaled_scores = raw_scores / math.sqrt(embedding_size)
```
 converting the scaled scores into probabilities
 `dim=1` ie, applies soft max across each row
 Every row become a probability distribution whos values add up to **1**

 ```python
 attention_weights=torch.softmax(scaled_scores, dim=-1)
 ```

 verify the result

 `dim=1`sum each row.
 every row should add up to **1** confirming valid probability distribution.

 ```python
 attention_weights.Sum(dim=-1)
 ```

 ###Data flow

 ```
 queries + keys => Raw Attention Scores => Divide by embedding size => scaled Attentionscore => softmax =>attentionWeights
 ```
 =======================================================================================

 Lesson: 13 Context vectors

 ##objectives

 understand how Attention weights are used to combine value vectors and create context vectors, giving each word a richer understanding of sentence.

 ###What i learned:

 + Every word has a **attention weight** for every other word
 + Every word also has a value vector containing information on what it can share.
 + The attention weights  decide how much of each value vector should be used.
 + combining the value vectors according to these weights creates the **context vector**
 + context vector is the final outcome  of the attention mechanism and contains the information gathered from the entire sentence.

 ---

 ### code decodeing
 -
 defining size as `embedding_size=4`

 creatinng embeddings as `embeddings= torch.rand((len(words), embedding_size))`

 then creating weight matrices used to generate queries, keys and values.
 `query_weight=torch.rand(embedding_size, embedding_size)` similary method for creating key_weight nad value_weight..

 qkv are crafted using associated reciepie weights..
 ie
 `q = embedding @ query_weights` similarly `k` and `v` also created by `@`(matrixmultiplication) respective weights.

 attention scores are generated by the formulae:	`q@k.T`

 then raw_scores are scaled by dividing by raw_score by the root of embedding_size..
 `scaled_scores = raw_scores/math.sqrt(embedding_size)`

 next we are using soft max from torch to make the scores of a row to add up to **1**
 `attention_weights= torch.softmax(scaled_scores, dim=-1)`

 context vectors are created by matrix multiplying this **attention_weights** and **values**.

 ```
 context_vector = attention_weights @ values
 ```

 This result is a new representation for every word that contains information from entire sentence.

 ###Data flow

 ```
 embeddings=> query_weights; queries , key_weights; keys, value_weights; values => raw_scores(attentionscores)=>scaled=> attentionweight=>weighted summ of value vectors => context vectors
 ```


 =======================================================================================

 Lesson 14: Single AttentionHead

 ###Objective:
  combine the complete attentionprocess into one reusable pytorch module.


  What i Learned:

  = `nn.Module` from torch is used to create reusable neural-network components.
  = A single attentionhead contains separate query , key and value layers.
  = `nn.Linear` replaces the manually created wweight matrices we did on previous lessons.
  = The `forward()` method defines how data moves through the attentino head.
  = calling the module automatically runs its `forward()`
  = the module return s both `contextvectors` and `attentionweights`

  --
  code decodeign..
  -

  importing neuralnetwork

  `import torch.nn as nn`

  creating custom neuralnetwork module. `nn.Module` is the base class for PyTorch models and reusable layers.

  `class SingleAttentionHead(nn.Module):`

initiating the attention Head
- `self` refers to current object, `embedding_size:int` expects embedding size to be an integer.
- `->None` means this constructor does not return a value.

`def __init__(self, embedding_size: int) -> None:`


initiating the parent` nn.Module ` class. so that it allows pytorch to track the layers and their trainable parameters..

`super().__init__()`


creating query transformation layer:
- `in_features` is the number of values entering the layer.
- `out_features` is the number of values produced by the layer.
- `bias=False` means no added bias value.

the key and value layers work in same way only difference is that they learn from different weights..

```python
self.query_layer = nn.Linear(
in_features = embedding_size,
out_features= embedding_size,
bias = False
)
```
---

`forward()` is the one defining how the **data/embeddings** move through the attention head.
`embeddings: torch.Tensor` here means input must be a Pytorch Tensor.
-`tuple[torch.Tensor, torch.Tensor]`return s 2 tensors..

```python
def forward(
	self,
	embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
```

transforming the same embedding into Q, K and V
```
queries = self.query_layer(embeddings)
keys = self.key_layer(embeddings)
values = self.value_layer(embeddings)
```

next attenttion scores here we call it scores ..
same formulae 
`transpose(dim0=-2, dim1=-1)` swaps the last two dimensionsof the key tensor.
this swapping allow every query to be compared with every key..

```
scores = queries @ keys.transpose(dim0=-2, dim1=-1)
```
next is scaling scores 

```
scaled_scores = scores/math.sqrt(self.embedding_size)
```

next soft max for converting to probabilities ..
`dim=-1` applies soft max acorss the final dimension and each row adds upto make **1**

```python
attention_weights= torch.softmax(
	scaled_scores,
	dim=-1
)
```

matrix multiplyiong this probabilities with values to get context 

```
context_vectors = attention_weights@values
```
and then returning both attention_weights and context_vectors..


creating one instance of attentionhead module.
```
attention_head=SingleAttentionHead(
	embedding_size=embedding_size
)
```

passing the embedding through the module instance.
```
context_vectors, attention_weights = attention_head(embeddings)
```


then pytorch automatically calls :

```
attention_head.forward(embeddings)
```

 ###Data flow

 ```
 embeddings=> query_layer; queries , key_layer; keys, value_layer; values => raw_scores(attentionscores)=>scaled=> attentionweight=> context vectors
 ```

======================================================================================================================

Lessson: 15 Transformer Block

###Objective:
understand how self-attention , residual connections, normalization and a feed-forward network are combined into  one Transformer block.


##What i learned:

- A transformer has two main processing stages:
	- **self_attention**
	- **feed_forward_network**
- Residual connections preserve  the original information  while adding newly learned information.
- Layer normalization keeps values stable as they move through the block.
- The input and output shapes remain the same, but the token representations will become more context aware.

---

###part1: self attention:

creating the query transformation layer
- `in_features` and `out_features` are taking `embedding_size`
- `bias` is set to `False`

```python
self.query_layer = nn.Linear(
	in_features=embedding_size,
	out_features= embedding_size,
	bias=False
)
```

Compares every query with every key.

- `transpose(dim0=-2, dim1=-2)` swaps the final two dimensions of the key tensor. the resulting will be attention score matrix.

```
scores  = queries @ key .transpose(dim0=-2, dim1=-1)
```

next scaling 
```
scaled_scores = scores/math.sqrt(self.embedding_size)
```

converting scaling scores to probabilities by using soft max
dim=-1 is for applying soft max to each row..
ie each word or token.
```
attention_weights= torch.softmax(
	scaled_scores,dim=-1)
```

now context_vectors

```
context_vectors= attention_weights @ values
```


###part 2:	Feed forward network:

expanding the hidden representation ie just like unfolding a paper..

```
expanded_size = embedding_size * 4
```

---

processes each token independently.
 the process is let embedding size is 4 
 then
 epanded size will be 16
 so ,

 first layer takes in embedding size  values and output will be expanded size

 relu() takes this expanded _size embedding and 
 

 ReLU stands for rectified Linear unit.. this is an activation function. After linear layer produces expanded outputs

 ReLU decides which values should continue forward..

 rule is very simple positive values stay and negetive values will be replaced by  `0` at respective positions.



 then the next linear layer take in expanded_size embedding given by ReLU and puts out embedding_size values.

 `` basically ReLU  is cutting out all the negetivity and spreading positivity``
 
 ```python
 
 self.network = nn.Sequential(
	nn.Linear(
		in_features = embedding_size,
		out_features= expanded_size
	),
	ReLU(),
	nn.Liner(
		in_features = expanded_size,
		out_features= embedding_size
	)
)

```

next snippet def forward()  is defining how data passes through the feed forward network.

- `embeddings : torch.Tensor` receive the token representation from the transformer block.
- `self.network(embeddings)`  passes  the embeddings through the layers stored inside the `nn.Sequential`.
- `torch.Tensor` indicates the output  returns a tensor.

##Part3: Residual Connection

Adds the original embeddings back to the the attention output.
This preserves the original token information while adding the contextual information.

```python
after_attention = embeddings + attention_output
```

###Part 4: layer normalization.

normalize the values across each tokens embeddings dimensions.
`normalized_attention=embedding_size`
ie, normalizing the final embedding dimension'

```python
self.first_layer_norm=  nn.LayerNorm(
	normalized_shape = embedding_size
)
```

stabilizing the result after the first residual connection.

```
normalized_attention = self.first_layer_norm(after_attention)
```

###part 5: Second Residual Connection

Passes the attention-enriched representation throught the feed forward network

```
feed_forward_output = self.feed_forward(normalized_attention)
```

preserving the attention result while adding the feed forward transformation.

```python
after_feed_forward = (
normalized_attention + feed_forward_output)
```

normalizeing the final output of the transformer block


```python
block_output= self.second_layer_norm(after_feed_forward)
```

production level this transformer code can be enhanced by adding :
= > true multihead attention,
= > output projections
= > causal masking for decoder models
= > Dropout
= > GELU, SiLU, SWiGLU activations
= > RMS Norm
= > Pre-Norm architecture...

==========================================================

