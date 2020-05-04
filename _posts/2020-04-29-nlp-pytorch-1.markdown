---
layout: post
title: "NLP with PyTorch : Tokenization and Embeddings"
date: 2020-05-03
description: A step-by-step guide to tokenize and create embeddings usable by PyTorch. # Add post description (optional)
img:  # Add image post (optional)
url: blog/nlp-pytorch-embeddings
---

### What's this about?

Natural Language processsing in the method of analyzing textual data and build intelligent systems that take advantage of abundant data available. This post whilst being self-sufficient, is part of a series of posts on an NLP problem. This series follows the step-by-step procedure that I followed in building a NLP system to tackle the problem of inference validation using SNLI dataset. In this post, the basic step to any NLP problem is discussed along with code for implementing this step using PyTorch, NLTK and Pandas.

Code available on [github](https://github.com/amdsrinivas/Blog-Codes).

---
### The Problem Statement
Inference Validation problem in NLP is to identify whether a given pair of statements support each other or contradict each other. The idea is to given two statements, say **text** and **hypothesis**, the system should be able to recognize one of the following :
- text *entails* hypothesis.
- text *contradicts* hypothesis.
- text and hypothesis are *neutral* to each other.

A well known dataset for inference validation is [**Stanford Natural Language Inference**](https://nlp.stanford.edu/projects/snli/) dataset.

This problem is a 3-way classification problem given *text* and *hypothesis* as features. The process of converting the textual data to an efficient interpretable representation for a model is the preprocessing step in any NLP problem. Let us dive right in!

------
### Cleaning and Exploring the data

Stanford being Stanford provides three sets of data without us worrying too much about splitting the data for training and testing purposes. But, due to the large number of training examples, one can use only the *training* set and setup *validation* and *test* sets from it. The results would not be comparable to any of the published works, but would avoid any computational hindrance right away.

The training set consists of 550152 sentence pairs that belong to 4 different categories. 
```python
INPUT_FILE = '<Train file path>'
COLUMNS = ['gold_label', 'sentence1', 'sentence2']
df = pd.read_json(INPUT_FILE, lines = True)
df = df[COLUMNS]
df.groupby('gold_label').count()
```

The above snippet of code would give the following output :

![Counts image]({{site.baseurl}}/assets/img/nlp_pytorch/class_counts.JPG)

The class "**-**" represents that for the given sentence pairs, the annotators could not achieve a majority. We can ignore those examples to simplify our problem. This gives us around 180000 sentence pairs for each class to work with.

------
### Tokenization using NLTK

In the tokenization step, an input sentence is converted into sequence of words. A general step during the tokenization process is to remove stopwords in the sentence. Stopwords are the set of words occur often in the language but do not add much value to the representation. They are words like "is", "are", "myself" etc., [Natural Language Tool Kit](http://www.nltk.org/) (NLTK) provides a wide range of tools for Natural language processing tasks. It provides a corpus for stopwords and a basic tokenization technique which can be used to implement a tokenizer. 

Tokenizer can be used to handle:
- punctuations
- out of vocabulary words
- stop words

In general, the output of a tokenizer is a sequence integer IDs for each word in the sequence. These integer IDs are used to keep track of the words. These IDs can be generated on the fly when proccessing the training corpus. But, when using pre-trained embeddings, the corresponding IDs of the vectors should be used when tokenizing the sequences.

> Other than tokens for words in vocabulary, two other tokens are required. one for "out-of-vocabulary" words and one for "padding" applied to the sequences.

The following demonstrates the use of nltk to clean and tokenize the data. It assumes the prior existence of vocabulary and lookup dictionary used. This is discussed in detail in the further sections.

```python

puncts = set([_t for _t in string.punctuation]) # import string
stop_words = set(stopwords.words('english')) # from nltk.corpus import stopwords
stop_words = stop_words.union(puncts)

def tokenize(sentence, sequence_length):
    tokens = []
    sentence = sentence.lower()
    pad_token = len(vocab) + 1
    for _tok in word_tokenize(sentence):
        if _tok not in stop_words:
            if _tok in vocab:
                tokens.append(word2idx[_tok])
            else:
                tokens.append(len(vocab))
    tokens = tokens + [pad_token for i in range(sequence_length-len(tokens))]
    return tokens[:sequence_length]
```

The above takes a sentence as input, removes the stopwords and punctuations and converts the tokens into token IDs. Such tokenized sequences are then padded with a padding token to chosen length to make sure all the input samples are of the same length.

------
### Embeddings in PyTorch

Embeddings is a technique used in NLP to represent words in a vector space so that related words can be placed accordingly in the vector space. Embeddings in any framework are generally implemented as a lookup store based on token ID. So, for example if a sequnce of length **m** is being processed and a embedding of dimension **n**, the entire sequence is fed to the model in a **m X n** matrix representation.

Embedding layer can be used in a torch model by using [*torch.nn.Embedding*](https://pytorch.org/docs/stable/nn.html#embedding) layer. So, this layer is incorporated in a model as follows :

```python
# Assuming torch.nn is improted as nn ( import torch.nn as nn )
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.embedding = nn.Embedding(5, 10) # ( vocab_size, embedding_dimension )
    
    def forward(self, inputs):
        return self.embedding(inputs)

_in = torch.tensor([[1,2,3], [0,4,3]])
print('Inputs shape : ', _in.shape)
_out = model()(_in)
print('Outputs shape : ', _out.shape)
```

The above demonstrates inputs of sequence length of 3. '*_in*' is a batch of size 2 with sequences of length 3. This gets transformed into a matrix of size *batch_size* X *sequence_length* X *embedding_dimension*. The output of the above code is as shown below :

![Counts image]({{site.baseurl}}/assets/img/nlp_pytorch/embeddings_output.JPG)

------
### Using pre-trained Embeddings : GLOve

A point to consider when using vanilla embedding layers is that, the embeddings have to learned during the training process. This will not always lead to better results as it would take a lot of data for training. In order to overcome this issue, pre-trained embeddings can be used. PTEs are trained using abundance of data available. It is a kind of transfer learning that can be employed for better interpretation of the data. In this post, Global Vectors ( [GLOve](https://nlp.stanford.edu/projects/glove/) ) are used. Stanford NLP provides different versions of embeddings trained on datasets of different magnitudes. One can choose a version of embeddings based on the computation power available. In this post, 6B.50d vectors are used. It has 6 billion tokens represented in 50 dimensions. 

In order to use the embedding provided, it has to be converted to a matrix form and vocabulary has to be setup. Each line in the embeddings provided by Stanford is of the form :

> WORD{space}[comma separated vector]

A generic processing method can be something like :

```python
def process_embeddings(path):
    vocab = []
    idx = 0
    lookup = {}
    vectors = []
    with open(path, 'rb') as f:
        for l in f:
            try:
                line = l.decode().split()
                word = line[0]
                vect = np.array(line[1:]).astype(np.float32) # import numpy as np
                vocab.append(word)
                vectors.append(vect)
                lookup[word] = idx
                idx += 1
                
            except Exception as e:
                print(e)    
    return vocab, lookup, vectors
```

In the above code, we process the embeddings line by line and build our vocabulary (**vocab**), lookup dictionary (**lookup**) and the vectors. vocabulary and lookup dictionary are used in tokenizing the sequences. Vectors are used in building the embedding matrix.

Embedding matrix is nothing but a simple *Vocabulary size* X *Embedding dimension* matrix which contains the vectors that can be indexed by the lookup dictionary using words in the vocabulary. The detailed code for building an embedding matrix is discussed in the next section after dealing with the *out of vocabulary* words and *padding* tokens.

------
### Handling padding and unknown tokens

Not always in our data do we encounter words that are in the vocabulary. There will be some rare words that do not have an embedding available. This is the case even when we do not use pre-trained embeddings and build a custom vocabulary. Rare words should not be included in the vocabulary because the amount of data needed to train the embeddings will be very minimal. In order to avoid such issues, rare words are represented by a single token called as "**Unknown tokens**" (UNK). Every word that does not belong to the vocabulary is represented by such a token by the tokenizer.

Next in line is the padding token. Pytorch requires all input sequences to the model to be of same length. But, in general, input sequences will almost never be of same length. In order to handle such cases, the sequences are padded using a "**padding token**". The padding token is added to a tokenized sequence to make all sequnces of equal length. In general, the embedding to a padding token is all zeroes so that the network gets a zero input and the weights are not affected by the padding tokens.

The following code is used to build an embedding matrix that handles both UNK tokens and padding tokens:

```python
def build_embedding_matrix(vocab, lookup, vectors):
    num_embeddings = len(vocab) + 2
    embedding_dim = len(vectors[0])
    weights_matrix = np.zeros( (num_embeddings, embedding_dim) )
    unknown_index = len(vocab)
    padding_index = unknown_index + 1
    for word in vocab:
        index = lookup[word]
        weights_matrix[index] = vectors[index]
    weights_matrix[unknown_index] = np.random.normal(scale=0.6, size=(embedding_dim, ))
    weights_matrix[padding_index] = np.zeros( (embedding_dim,))
    print(weights_matrix.shape)
    return weights_matrix
```

In the above code, *unknown_index* and *padding_index* are used to track the UNK and padding tokens respectively. The embedding corresponding to an unknown token is a uniform normal distribution with a standard deviation of 0.6. This is can be any data distribution as it effects the training process in a minor scale.


------
### Stitching it all together

A model that uses a pre-trained embedding layer needs few modifications from the above code. The embedding matrix should be loaded as weights to the layer and based on out choice of further training, these embeddings can be freezed. The following code shows the necessary modifications to the model.

```python
# Assuming torch.nn is improted as nn ( import torch.nn as nn )
class model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # ( vocab_size, embedding_dimension )
        self.embedding.load_state_dict({'weight': torch.tensor(embedding_weights_matrix, dtype=torch.float64)})
        self.embedding.requires_grad = False
    
    def forward(self, inputs):
        return self.embedding(inputs)

_in = torch.tensor([[1,2,3], [0,4,3]])
print('Inputs shape : ', _in.shape)
_out = model()(_in)
print('Outputs shape : ', _out.shape)
```

The *load_state_dict* method loads a torch tensor containing the pre-trained embeddings into the embedding layer. The flag *requires_grad* determines whether these loaded embeddings are to be updated over the course of training. For this experiment, the embedding layer is freezed.

The output of the above code is as shown below:

![Counts image]({{site.baseurl}}/assets/img/nlp_pytorch/glove_embeddings_out.JPG)

------
### Conclusion

In this post, a basic step-by-step procedure to get started with a NLP problem is dicussed. Tokenization using NLTK is elaborated and the use of embeddings in a pytorch model is demonstrated. Procedure to use a pre-trained embedding layer is presented. In the next post, a model to solve the NLP problem will be discussed along with code samples. The code for this blog is available on [github](https://github.com/amdsrinivas/Blog-Codes).