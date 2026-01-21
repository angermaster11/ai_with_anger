# Introduction to PyTorch by Arju Srivastava
This repository contains a comprehensive introduction to PyTorch, a popular deep learning framework. The materials are designed to help beginners get started with PyTorch and understand its core concepts and functionalities.

## What is PyTorch?
PyTorch is a Python-based scientific computing package serving two broad purposes:
- A replacement for NumPy to use the power of GPUs and other accelerators.
- An automatic differentiation library that is useful to implement neural networks. 
- (Iska mtlb yeh hai ki yeh humko neural networks banane mein madad karta hai aur yeh GPU ka use karke computations ko fast karta hai)

### Goal of this tutorial:
Understand PyTorch’s Tensor library and neural networks at a high level.
Train a small neural network to classify images

## What are Tensors?
Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other specialized hardware to accelerate computing. If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along in this quick API walkthrough.

```python
import torch
import numpy as np
```

### Tensor Initialization
Tensors can be initialized in various ways. Here are some common methods:

#### Directly from data
Tensors can be created directly from data. The data type is automatically inferred.

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

#### From a NumPy array
Tensors can be created from NumPy arrays. The resulting tensor and the NumPy array share the same memory location.
```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```
#### From another tensor
The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
```python
x_ones = torch.ones_like(x_data) # retains the shape of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype to float
``` 
#### With random or constant values:
Shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```
Output:
```tensor([[0.1234, 0.5678, 0.9101],
        [0.1121, 0.3141, 0.5161]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```
### Tensor Attributes
Tensor attributes describe their shape, datatype, and the device on which they are stored.
```python
tensor = torch.rand(3,4)
print("Shape of tensor:", tensor.shape)
print("Datatype of tensor:", tensor.dtype)
print("Device tensor is stored on:", tensor.device)
```
Output:
```Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```
## Tensor Operations
Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, slicing, etc.), sampling and more are comprehensively described in the [documentation](https://pytorch.org/docs/stable/torch.html).

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
  print(f"GPU Name: {torch.cuda.get_device_name(0)}")
  print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
```
Output:
```Device tensor is stored on: cuda:0
```
#### Standard numpy-like indexing and slicing:
```python
tensor = torch.ones(4, 4)
print(tensor)
tensor[:,1] = 0
print(tensor)
```
Output:
```tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
#### Joining tensors
You can use torch.cat to concatenate a sequence of tensors along a given dimension. See also torch.stack, another tensor joining op that is subtly different from torch.cat.

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```
Output:
```tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

#### Why String converts into Numbers ?
When you input text into a machine learning model, it needs to be converted into a numerical format that the model can understand and process. This is typically done through a process called tokenization, where the text is broken down into smaller units (tokens) such as words or subwords, and each token is then mapped to a unique numerical identifier (usually an integer). These numerical representations allow the model to perform mathematical operations on the data, enabling it to learn patterns and make predictions based on the input text.
## Conclusion
This introduction covered the basics of PyTorch, including tensor creation, attributes, and operations. With this foundation, you can start exploring more advanced topics in PyTorch, such as building and training neural networks. Happy coding!

#### Can I use these tensors as embeddings for text data?
Yes, tensors can be used as embeddings for text data. In natural language processing (NLP), embeddings are dense vector representations of words or phrases that capture their semantic meaning. PyTorch provides various tools and libraries, such as TorchText and Hugging Face's Transformers, that facilitate the creation and use of embeddings in text-based models.

#### Compare PyTorch Tensor with Word2Vec Embeddings && Transformers Embeddings
PyTorch Tensors are a general-purpose data structure used for numerical computations, while Word2Vec and Transformer embeddings are specific types of vector representations used in natural language processing (NLP).
- PyTorch Tensors: These are multi-dimensional arrays that can hold data of various types (e.g., floats, integers). They are used for a wide range of applications, including image processing, time series analysis, and NLP. Tensors can be manipulated using various operations provided by the PyTorch library.
- Word2Vec Embeddings: Word2Vec is a specific technique for generating word embeddings. It uses a shallow neural network to learn vector representations of words based on their context in a large corpus of text. The resulting embeddings capture semantic relationships between words, allowing similar words to have similar vector representations.
- Transformer Embeddings: Transformers, such as BERT and GPT, use a more complex architecture to generate contextual embeddings. Unlike Word2Vec, which produces static embeddings, Transformer models generate dynamic embeddings that take into account the surrounding context of a word in a sentence. This allows for a deeper understanding of word meaning based on usage.

## Text Generation using Transformers and PyTorch
This notebook demonstrates how to use the Hugging Face Transformers library along with PyTorch to generate text based on a given prompt. It includes setting up the environment, checking for GPU availability, and defining a function to generate text using a pre-trained model.

#### Multiplying Tensors

```python
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
```
Output:
```tensor.mul(tensor)
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
tensor * tensor
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
#### Matrix Multiplication
```python
import torch
# Create two tensors
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
# Perform matrix multiplication
result = torch.matmul(a, b)
print(result)
```
Output:
```tensor([[19, 22],
        [43, 50]])
```

#### Inplace Operations
Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.
```python
# Inplace operations are operations that directly modify the content of a tensor without making a copy.
tensor = torch.ones(5)
print(tensor)
tensor.add_(5)
print(tensor)
```
Output:
```tensor([1., 1., 1., 1., 1.])
tensor([6., 6., 6., 6., 6.])
```
## Note
- In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

## Bridge with NumPy
Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

### Tensor to NumPy array
```python
t = torch.ones(5)
print(t)
n = t.numpy()
print(n)
```
Output:
```tensor([1., 1., 1., 1., 1.])
array([1., 1., 1., 1., 1.])
```
A change in the tensor reflects in the NumPy array.
```python
t.add_(1)
print(t)
print(n)
```
Output:
```tensor([2., 2., 2., 2., 2.])
array([2., 2., 2., 2., 2.])
```

### NumPy array to Tensor
```python
n = np.ones(5)
print(n)
t = torch.from_numpy(n)
print(t)
```
Output:
```array([1., 1., 1., 1., 1.])
tensor([1., 1., 1., 1., 1.])
``` 
