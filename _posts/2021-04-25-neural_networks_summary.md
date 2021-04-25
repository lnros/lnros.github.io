---
title: "Neural Networks: a technical summary"
date: 2021-04-25
categories:
  - machine-learning
tags:
  - neural-networks
---

The focus of this summary is in the main idea regarding neural networks (NN), so proofs were not contemplated and other mathematical explanations were not described in depth or were skipped whatsoever. The summary is based on the work of Shalev-Shwartz and Ben-David [[1]](#Reference) and is divided as follows: Section [1](#sec:feed_nn) introduces the structure of NN and Section [2](#sec:learn_nn) brings very concisely the learning terms into the NN world. Section [3](#sec:exp_pwr) regards what can actually be implemented and it was not included in its full extent. In Section [4](#sec:vcdim), I present briefly in words the idea behind the proof of the size the VC dimension of a hypothesis class. The complexity of learning a NN is brought in short in Section [5](#sec:runtime). Section [6](#sec:sgd) presents the use of Stochastic Gradient Descent (SGD) with backpropagation and it does not contain the pseudocodes for the algorithms and I advise that those interested in further reading to check out the [Reference](#Reference) and the [Further Reading](#Further_Reading) sections.

## 1. Feedforward Neural Networks <a name="sec:feed_nn"></a>

The ultimate objective of a NN is to perform complex computations. One can understand its structure as a graph that connects its nodes with directed edges in such a way that each output of a node is the input of the next one. The main remark here is that I define each node as a neuron.

Assume we have a directed acyclic graph $G = (V,E)$, where the set $V$ represent the neurons and the set $E$ the edges. I represent a feedforward NN by $G$ together with a weight function over the edges, described as: $w: E \rightarrow \mathbb{R}$. Besides, every neuron has an activation function $\sigma$ that can take some different forms (such as the sign, the sigmoid  and the threshold functions). The information feeded to the neuron is given by the weighted sum based on $w$ provided by all the other neurons that are connected to this one.

It is common to assume that the network is structured in layers, where the neurons are organized in a union of disjoint subsets in a way that every edge links a node from a previous layer to another neuron in the subsequent layer. The first layer ($V_0$) is called the input layer and it has $n+1$ neurons, given that $n$ is the dimensionality of the input space. The remainder neuron in this layer is a constant that always outputs 1. Every neuron is represented by $v_{t,i}$, where $t$ indicated the layer and $i$ the number of the neuron in this specific layer. Every neuron produces an output $o_{t,i}(x)$, and $x$ is the input vector being fed into the network.

In order to calculate layer by layer, I now assume that we have the input provided from the previous layer. We shall denote the input of $v_{t+1,j}$ as $a_{t+1,j}(x)$. Then, 


$$a_{t+1,j}(x) = \sum_{r:(v_{t,r},v_{t+1,j}) \in E} w((v_{t,r},v_{t+1,j}))o_{t,r}(x)$$


$$
o_{t+1,j}(x) = \sigma(a_{t+1,j}(x))
$$


If $T$ is the total number of layers, $V_1,\dots,V_{T-1}$ are hidden layers and we call $V_T$ the output layer, which only has one neuron that is the output for the entire network. The network size is $\mid V \mid$, its depth is given by $T$, whereas its width by $\max_t\mid V_t\mid$.

![feedforward](https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif)

*In a feed forward network information always moves one direction; it never goes backwards. Source: [Paskari][ff-img]*


## 2. Learning Neural Networks <a name="sec:learn_nn"></a>
A hypothesis class for a NN is any set of functions such as: $h_{V,E,\sigma,w}: \mathbb{R}^{\mid V_0 \mid -1}\rightarrow \mathbb{R}^{\mid V_T \mid}$. The learning is usually performed by first fixing a graph and an activation function, and then searching which weight (parameter) best describes the data.

## 3. The Expressive Power of Neural Networks <a name="sec:exp_pwr"></a>

Fix an architecture $(V,E,\sigma)$. Every Boolean function  $\{\pm1\}^n \rightarrow \{\pm 1\}$ can be implemented by NN. But the number of neurons in the hidden layers might grown exponentially.

 <!-- %This is in agreement with the following theorem: %\textbf{Theorem: } For every $n$, let $s(n)$ be the minimal integer such that there exists a graph $(V,E)$ with $\midV\mid = s(n)$ such that the hypothesis class $\mathcal{H}_{V,E,sign}$ contains all the functions from $\{0,1\}^n$ to $\{0,1\}$. Then, s(n) is exponential in n. Similar results hold for $\mathcal{H}_{V,E,\sigma}$ where $\sigma$ is the sigmoid function.
 -->
<!-- %This theorem is true for every activation function if the weight vector can be conveyed by a bounded number of bits. -->

We cannot convey all possible Boolean functions with a network of polynomial size. On the other hand, it is feasible to express all Boolean functions that are able to be computed in time $O(T(n))$ as a network of size $O(T(n)^2)$.

NN are universal approximators. This means that for all fixed precision parameters $\epsilon > 0$ and all Lipschitz functions $f: [-1,1]^n \rightarrow [-1,1]$, we can build a network with guaranteed precision. Formally, this means that the network's output is ranged in the interval $[f(x)-\epsilon,f(x)+\epsilon]$, for every input $x \in [-1,n]^n$. Nonetheless, the network's size cannot be polynomial in $n$.

## 4. The Sample Complexity of Neural Networks <a name="sec:vcdim"></a>

In order to compute the sample complexity of learning a class $$\mathcal{H}_{V,E,\sigma}$$ one has to find this class's VC dimension. For the sign activation function, the VC dimension depends on the number of parameters needed to learn. Thus, given that the model has to learn $\mid E \mid$ parameters (since the weights are a mapping from the edges to the real numbers), the VC dimension of $$\mathcal{H}_{V,E,sign}$$ is $$O(\mid E \mid \log(\mid E \mid))$$. The main idea behind the proof is the following: 


Let the hypothesis class be written as a composition of hypothesis classes for every layer. Hence, we have that the product of the growth functions for every class is the upper bound of the growth function of the composition. Also, let each hypothesis class for every layer be a product of function classes. Then, the growth function of every layer's hypothesis class is bounded by the product of the classes. Having that a neuron is a homogeneuos halfspace hypothesis and that the VC dimenstion of homogeneus halfspaces is the dimension of their input, we can apply Sauer's lemma. Finally, performing some calculation we arrive at the formal proof.


For the sigmoid activation function, the VC dimension has both lower and upper bounds. The first is the squared value of parameters and the second is the product of the square number of neurons and the square number of parameters. But, by discretization of the problem, given what happens in practice, the VC dimension is $O(\mid E \mid)$.

## 5. The Runtime of Learning Neural Networks <a name="sec:runtime"></a>

It is NP-hard to implement an Empirical Risk Minimization for a hypothesis class with the sign activation function. To go pass this, one might think that it does not have to reach an exact result, but one that yields a low empirical error. The problem with this approach is that finding the parameters for the approximate solution is computationally infeasible. 

Changing the architetcture of the network or the activation function is most likely to keep the hardness of the problem. That is why a heuristic approach it is usually used. It is based in the SGD algorithm.


## 6. SGD and Backpropagation <a name="sec:sgd"></a>

The goal of the SGD algorithm is to minimize the risk function of the network. For it to work, the activation function must be a differentiable scalar function. The weight function is a vector $w \in \mathbb{R}^{\mid E \mid}$. The function that the network computes is given by $h_w: \mathbb{R}^n \rightarrow \mathbb{R}^{k}$, where $n$ indicates the input neurons and $k$ the output ones. Let $\Delta(h_w(x),y)$ be the differentiable loss function of predicting $h_w(x)$. Then, for a distribution $\mathcal{D}$ over $\mathbb{R}^{n} \times \mathbb{R}^{k}$, we have that the risk of the network is:

$$
L_{\mathcal{D}}(w) = \mathbb{E}_{(x,y)\sim D} [\Delta(h_w(x),y)].
$$


Since the NN problem is not convex, some changes have to be made in the SGD:


- The parameters vector cannot be initialized to zero, but to random values close to zero instead so as to prevent the hidden neurons to receive the same weights. Repeating the SGD multiple times, with different random initializations, we ought to eventually find a good local minimum.
- We use a variable step size instead of a fixed one due to nonconvexity. The choice of the step is made empirically.
- The best vector is used in a validation set. Ocasionally, a regularization parameter is added to the weights.
- The SGD by itself does not yield a gradient and it has to be implemented using the backpropagation algorithm.


The goal is to find the gradient of the loss function on a particular sample given the weight vector. This is done by computing the partial derivatives of $w$ with respect to the edges. 

Recall that the set of nodes is decomposed into layers. For every layer there is a matrix that represents the potential edge between the neurons in two subsequent layers. If indeed there is this edge in $E$, the corresponding value in the matrix is assigned respecting $w$. If there is no such edge, a phantom edge is created and its assigned weight equals zero. These phantom edges have no impact when calculating the partial derivatives, which are computed using the Jacobian of the loss function of layers' subnetworks. To do this, one has to find the input and output vectors for each layers form the bottom up and then to find the gradient of the loss of each output from the top down.


#### Reference
<a name="Reference"></a>
[1] Shai Shalev-Shwartz, Shai Ben-David. "Neural Networks". In *Understanding Machine Learning: From Theory to Algorithms*, 262-276. New York: Cambridge University Press, 2014.

#### Further reading <a name="Further_Reading"></a>
[2] Shai Shalev-Shwartz, Shai Ben-David. *Understanding Machine Learning: From Theory to Algorithms*, 262-276. New York: Cambridge University Press, 2014.

[3] VC-Dimension on [Wikipedia][vc-dim]

[4] Stochastic Gradient Descent on [Wikipedia][sgd]

[5] Lipschitz continuity on [Wikipedia][lipschitz]

[6] Feedforward neural networks on [Wikipedia][ff-nn]

[7] Backpropagation on [Wikipedia][bp-nn]

[8] Activation functions on [Wikipedia][act-fun]

[8] Empirical Risk Minimization on [Wikipedia][erm]

[vc-dim]: https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension
[sgd]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
[lipschitz]: https://en.wikipedia.org/wiki/Lipschitz_continuity
[ff-nn]: https://en.wikipedia.org/wiki/Feedforward_neural_network
[ff-img]: https://en.wikipedia.org/wiki/File:Feed_forward_neural_net.gif
[bp-nn]: https://en.wikipedia.org/wiki/Backpropagation
[act-fun]: https://en.wikipedia.org/wiki/Activation_function
[erm]: https://en.wikipedia.org/wiki/Empirical_risk_minimization

