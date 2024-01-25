# Front-prop

_"Learn through inference"_.

Copyright © 2021 **Karol Pajak**

---

**All contents of this repository, including all ideas described here, are the intellectual property of the Author, and any usage or sharing of the ideas or the code without obtaining the Author's permission is strictly prohibited.**

---

This repository contains a Proof of Concept of the Front-prop algorithm.

See the [`frontprop_fast.ipynb`](https://github.com/rohrl/frontprop/blob/main/frontprop_fast.ipynb) notebook for a quick demo.

### Summary

Front-prop is a an unsupervised learning algorithm which addresses some of the limitations of backpropagation (see below). Due to unsupervised nature, it does not try to optimise for specific expected output or a specific task, bur rather aims to produce useful representations of inputs, regardless of the domain. Useful in the sense of capturing common patterns in the input data, which can be viewed as a way of (lossy) compression of the input data. As such there is no explicit training objective and learning achieved by continously fitting to patterns in the incoming data and producing efficient representations of these patterns (a.k.a. embeddings).

There is no separation between training and inference - there is only forward pass (hence "front prop") and each pass changes the network, to continously adapt it to incoming data - not unlike how organic brains are understood to work. 
Consequently there is also no stop condition, and therefore the algorithm should naturally handle distribution shifts **[unverified]**, which are present in real world environments.

Note that in most cases Front-prop should be easy to adopt to modern Neural Network architectures (e.g. CNNs, Transformers), just by replacing backpropagation, without compromising the model architecture **[unverified]**.
It should also be less compute/memory demanding, as it does not have to store gradient information, nor perform backward pass **[unverified]**.

### Results

TODO

### Motivation

The motivation behind Front-prop is to seek alternative learning algorithms to backpropagation, which are more similar to how humans and other intelligent beings learn, in order to find ones that are better suited for achieving AGI.

Backpropagation has been the foundation of Deep Learning and all its latest breakthroughs in AI. But it suffers from several constraints that make it unlikely to be the way how biological brains learn in nature and probably insufficient for AGI ([LeCun, 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf); [Hinton, 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf)). 

Some of the limitations of backpropagation are (not exhaustive):
1. strict separation between training phase and inference phase (sepearation in time, in data, and in kinds of operations),
1. necessity to either explicitely provide expected outputs, or a task/domain-specific loss function 
1. inability to continuously learn and infer at the same time
1. necessity to obtain all training data upfront, and before inference phase
1. distribution shifts between real data and training data
1. requires saving each layer's forward results to be used for the backward pass
1. computationally intensive 
1. requires knowledge of derivatives of each layer

Front-prop algorithm is partially inspired by the Hebbian theory ([Hebb, 1949](https://en.wikipedia.org/wiki/Organization_of_Behavior)) of how neurons in the brain operate and by the Perceptron model ([McCulloch, Pitts, 1943](https://www.bibsonomy.org/bibtex/13e8e0d06f376f3eb95af89d5a2f15957/schaul)).


### Detailed description

TODO.
(for now see implementation or contact the author)


---

Copyright © 2021 **Karol Pajak**
