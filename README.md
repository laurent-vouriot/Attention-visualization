# Attention 
Attention mechanism in neural networks aims to mimic human attention. It is usually defined using a Query, Key and Values. In a real life situation one could relate to queries keys and values with the following scenario. You are searching for a restaurant to have dinner, the attention mechanism would help you to focus on certain aspects of the restaurant (the values) based on your personal preferences (queries) and the information available (keys).

   - Queries could be the information you are looking for : italian, cheap, cosy.
   - Keys could be the information about each restaurant helping you to make a decision.
   - Values being the actual restaurant, the name for example.
  
The goal of attention mechanism is to compute a similarity between your query and the keys, in order to weight each values (restaurants), allowing to prioritize the restaurant matching your request as much as possible.

## Some background 

Attention pooling concept can be attributed to Nadaraya–Watson kernel Regression. 
Nadaraya and Watson proposed to estimate $m$ as a locally weighted average, using a kernel as a weighting function : 

```math
\begin{align}
    f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{ \sum_{j=1}^n K(x - x_j)}y_i.
\end{align}
```
K being a smoothing kernel. We can rewrite the above as follow : 
```math 
\begin{align}
    f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i

    \alpha(x, x_i) \triangleq  \frac{K(x - x_i)}{ \sum_{j=1}^n K(x - x_j)}y_i
    
\end{align}
```
To display that the prediction is thus a weighted sum of the outputs at the training point, where the weights depends on the similarity of the key $x$ and the query $x_i$.  


Let's use a gaussian Kernel 
```math
\begin{align}
    K(u) = \frac{1}{\sqrt{2\pi}}exp(-\frac{u^2}{2}).
\end{align}
```
Introducing the gaussian kernel (RBF) in the previous equation : 
```math
\begin{align}
    f(x) & = \sum_{i=1}^n \alpha(x, x_i) y_i \\
         & = \sum_{i=1}^n \frac{exp(-\frac{1}{2}(x - x_i)^2)}{\sum_{j=1}^n exp(-\frac{1}{2}(x-x_i)^2)}y_i \\
         & = \sum_{i=1}^n softmax\left( -\frac{1}{2}(x - x_i)^2 \right)y_i. 
\end{align}
```

We end up with a computation of the similarity between the query and the key $-\frac{1}{2}(x - x_i)^2 $. We end up with a softmax function that returns a weight distribution. In this context the label $y_i$ is the value (according to the previous analogy, the restaurant). 


## Most common attention mechanismes : 

*Additive attention* (Bahdanau et al.) : 
```math
\begin{align} 
    f_{att}(q, K, V) = softmax(w^t_v tanh( W_q q + W_k K)) v
\end{align}  
```

*Scaled Dot Product attention* (Vaswani et al.) : 
```math
\begin{align}
    f_{att}(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}})V
\end{align}
```

### Scaled dot product is an RBF kernel : 

From [Attention is Kernel Trick Reloaded] : 

Let $\alpha_{ij}$ be $\frac{1}{Z_1(a)}exp\left(    \frac{q_i k_j^T}{\sqrt{d_k}} \right)$ where $Z_1(a)$ is a normalizing constant. i.e $\alpha_{ij}$ is out attention scoring function. Then $\alpha_{ij}$  has the form : 
```math
\begin{align}
    \frac{1}{Z_1(a)}exp\left(    \frac{- \| q_i - k_j \|^2_2}{2 \sqrt{d_k}} \right) \times exp\left( \frac{\|q_i\|_2^2 + \|k_j\|_2^2}{2 \sqrt{d_k}} \right)  \\
\end{align}
``` 

where : 
```math
\begin{align}
    \|q_i - k_j \|_2^2 = \|q_i\|_2^2 + \| k_j \|^2_2 - 2q_ik_j^T
\end{align}
```

$exp\left(    \frac{- \| q_i - k_j \|^2_2}{2 \sqrt{d_k}} \right)$  is an RBF kernel distance and $exp\left( \frac{\|q_i\|_2^2 + \|k_j\|_2^2}{2 \sqrt{d_k}} \right)$ is a magnitude term which weights each query-key pair. 











