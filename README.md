# Attention 
Attention mechanism in neural networks aims to mimic human attention. It is usually defined using a Query, Key and Values. In a real life situation one could relate to queries keys and values with the following scenario. You are searching for a restaurant to have dinner, the attention mechanism would help you to focus on certain aspects of the restaurant (the values) based on your personal preferences (queries) and the information available (keys).

   - Queries could be the information you are looking for : italian, cheap, cosy.
   - Keys could be the information about each restaurant helping you to make a decision.
   - Values being the actual restaurant, the name for example.
  
The goal of attention mechanism is to compute a similarity between your query and the keys, in order to weight each values (restaurants), allowing to prioritize the restaurant matching your request as much as possible.

## Some background 

First Attention pooling idea can be attributed to Nadaraya–Watson kernel Regression. 
Nadaraya and Watson proposed to estimate $m$ as a locally weighted average, using a kernel as a weighting function : 

```math
\begin{align}
    f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{ \sum_{j=1}^n K(x - x_j)}y_i.
\end{align}
```
K being a kernel. In a more attention based perspective it can be written as : 
```math 
\begin{align}
    f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i
\end{align}
```
$x$ being the keys, $x_i$ the query and $v$ the values and a the similarity function. 

Let's use a gaussian Kernel 
```math
\begin{align}
    K(u) = \frac{1}{\sqrt{2\pi}}exp(-\frac{u^2}{2}).
\end{align}
```
Introducing the gaussian kernel in the previous equation : 
```math
\begin{align}
    f(x) & = \sum_{i=1}^n \alpha(x, x_i) y_i \\
         & = \sum_{i=1}^n \frac{exp(-\frac{1}{2}(x - x_i)^2)}{\sum_{j=1}^n exp(-\frac{1}{2}(x-x_i)^2)}y_i \\
         & = \sum_{i=1}^n softmax\left( -\frac{1}{2}(x - x_i)^2 \right)y_i. 
\end{align}
```

We end up with a computation of the similarity between the query and the key $-\frac{1}{2}(x - x_i)^2 $. Plugged into the softmax function to get a weight distribution. The label $y_i$ is the value. This derivation is just to display that kernel regression can be seen as a precursor of the modern attention formalism. 

Indeed most common attention scoring function can be formulated in the following form 

*Additive attention* (Bahdanau et al.) : 
```math
\begin{align} 
    \alpha(q, k) = w^t_v tanh( W_q q + W_k K) 
\end{align}  
```

*Scaled Dot Product attention*
```math
\begin{align}
    \alpha(Q,K,V) = \frac{QK^t}{\sqrt{d}}
\end{align}
```
