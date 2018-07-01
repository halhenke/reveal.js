## Pytorch
---
## Deep Learning Frameworks in 2018
- There are currently several frameworks with *essentially* the same capabilities/structure
    - Tensorflow (Google)
    - MXNet (Apache/Amazon (sort of))
    - CNTK (Microsoft)
    - Caffe2 (Facebook)
    - PaddlePaddle (Baidu)
    - Neon (Intel)
---
## Deep Learning Frameworks in 2018

- some differences but mostly
    - based on the same ideas
    - based around the same "primitives" (tensors, layers, graphs)
    - come with built in implementations of the standard set of DL architectures
        - CNNs
        - RNN/LSTM
        - etc
---
## Static Computational Graphs

- Vast majority of frameworks (Tensorflow etc) are based around constructing Static Computational Graphs
- Build a graph of computational Operations
- Static Compilation
-
---
### Advantages
- Easier to Deploy
- Easier to Optimize
    - Can be statically analysed, pruned, split across devices etc
- Graph is language independent
    - Frontends in multiple languages to build/run tensorflow graphs
---
### Disadvantages
- Inflexible
    - Some things are very hard to express with a static graph
- e.g. Recurrent Layers/Networks

---
## Unrolled RNN

![](/images/unrolled-rnn.png)

---
## Unrolled RNN

- We unfold recurrent networks into fixed numbers of steps
- A recurrent layer is built to analyse a fixed/pre-determined sequence length (words, sentences, etc)
    - If a sample has less elements than the sequence length we pad it with empty tokens
    - If it has too many elements we clip it
---
## Enter Pytorch

...from Facebook

![](/images/zuckerberg-stare.gif)

---
## PyTorch - Cool features
- Numpy style tensors that can run on CPU or GPU
- Dynamic graph construction/execution
- Smaller API
- Python all the way
---
## Dynamic Computational Graph
- The Graph can change every time we run the model
    - even if it ends up the same it is rebuilt every time
    - e.g. a recurrent layer can be every bit as long/short as a given sequence
---
## Dynamic vs Static
- Dynamic
    - "Define by Run"
- Static
    - "Define THEN Run"
---
![](/images/dynamic_graph.gif)
---
## Dynamic Computational Graph - But How?
<!-- ![](/images/how-cat.gif) -->
![](/images/how-monkey.gif)
---
## Reverse Mode Auto-Differentiation
- Each tensor can store a history of how it was created
- i.e.
    - which operation/function (addition, multiplication)
    - which tensors it was constructed from
- Each tensor operation/function has a `forwards` & `backwards` method
    - normal computation and computation of the derivative
---
## Reverse Mode Auto-Differentiation
- as we step forwards through a model - from input to output - we are also constructing a graph
- Once we have the output/loss we can use this graph and the `backwards` method to calculate the gradient
  - we can trace operations from the outputs to the inputs to calculate gradient
  <!-- - So as we travel forwards each tensor is both computing output and building a graph that will be used to compute the gradient -->
<!-- - Every forward pass == a new graph -->
---
```python
# define t
>>> t = torch.ones([2, 2])
>>> print(t.grad_fn)
>>>
# y is constructed from t
>>> y = t + 2
>>> print(y.grad_fn)
>>> < AddBackward0 object at 0x7f12da7f6a20 >
# z is constructed from ty
>>> z = y * 3
>>> print(z.grad_fn)
>>> <MulBackward0 at 0x107bc6240>
# Begin to step through the graph
>>> z.grad_fn.next_functions
>>> ((<AddBackward1 at 0x107bc4dd8>, 0),)
```
---
## Why People Dig it
- More interactive experience than working with something like tensorflow
    - Instant evaluation/feedback of model building blocks
- Easier to try novel architectures/ideas
    - Quite a few research papers implemented in pytorch these days
- Smaller, more consistent API than tensorflow
---
## The Future for Pytorch
- Pytorch & Caffe2 are being merged as part of Pytorch 1.0 Roadmap
- Try to make the deployment/production side of things easier
- Ways to export Pytorch graphs that retain dynamic characteristics

---
## The Future for Deep Learning Frameworks in General
- Tensorflow has developed Tensorflow Eager mode
    - more or less official now (1.9)
- Convergence of two approaches

---
## Negatives
### A lot of Redundancy & Duplication

![](/images/ultron-call-ultrons.gif)

---
## Positives
### A Chance for Standards to Arise - ONNX

- Open Neural Network Exchange
- A standard format in which to represent/serialize a framework/machine independent symbolic graph

---
## ONNX

- Allows you to
    - build graph in MXNet
    - export to ONNX
    - load in Caffe2
    - change graph
    - export to ONNX
    - etc
---
## Eventually everyone is happy

![](/images/brad-pitt-sucks.gif)
---
# THE END
