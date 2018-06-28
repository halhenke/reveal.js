## Pytorch
---
## Modern Frameworks

- In 2018 there are several frameworks with essentially the same capabilities/structure
    - Tensorflow (Google)
    - MXNet (Apache/Amazon (sort of))
    - CNTK (Microsoft)
    - Caffe2 (Facebook)
    - PaddlePaddle (Baidu)

---
## Converging on a standard format - ONNX

- Open Neural Network Exchange
- A standard format describing a framework/machine independent symbolic graph
- Allows you to
    - build graph in MXNet
    - export to ONNX
    - load in Caffe2
    - change graph
    - export to ONNX
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
    - We unfold recurrent networks into fixed numbers of steps
    - A recurrent layer is built to analyse a fixed number of words/sentences
    - If a sample has less words than this we pad it with empty tokens
    - If it has too many words we clip it
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
- Dynamic
    - "Define by Run"
- Static
    - "Define THEN Run"
---
![](/images/dynamic_graph.gif)
---
## Dynamic Computational Graph - But How?
![](/images/how-monkey.gif)
---
## Reverse Mode Auto-Differentiation
- Each tensor can store a history of how it was created
    - which operation/function (addition, multiplication)
    - which tensors it was constructed from
---
## Reverse Mode Auto-Differentiation
- as we travel forwards through a model each layer is a function of previous layers/tensors
  - we can trace operations from the outputs to the inputs to calculate gradient
  - So as we travel forwards each tensor is both computing output and building a graph that will be used to compute the gradient
  - Every forward pass == a new graph
- Each tensor operation is defined with an inverse operation
- Therefore every tensor can easily calculate its gradient
---
```python
# Track history of this tensor with requires_grad flag
t = torch.ones([2, 2], requires_grad = True)
print(t.grad_fn)
>>>
y = t + 2
print(y.grad_fn)
>>> < AddBackward0 object at 0x7f12da7f6a20 >
z = y * 3
print(z.grad_fn)
>>> <MulBackward0 at 0x107bc6240>
z.grad_fn.next_functions
>>> ((<AddBackward1 at 0x107bc4dd8>, 0),)
# # Calculate Gradient:
# z.backwards()# Gradient is now stored in t.grad
# print(z.grad)
```
---
## Why People Like it
- Feels easier to play around with than tensorflow
    - Instant evaluation/feedback of model building blocks
    - quite a few research papers implemented in pytorch
- Smaller, more consistent API than tensorflow
---
## The Future
- Pytorch & Caffe2 are being merged as part of Pytorch 1.0 Roadmap
- Try to make the deployment/production side of things easier
- Ways to export Graph to ONNX without ditching dynamism
## The Future
- Tensorflow has developed Tensorflow Eager mode
    - more or less official now (1.9)
- Convergence of two approaches
