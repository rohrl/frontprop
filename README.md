# Front-prop

_"Learn through inference"_.

Copyright © 2021 **Karol Pajak**

---

**All contents of this repository, including all ideas described here, are the intellectual property of the Author, and any usage or sharing of the ideas or the code without obtaining the Author's permission is strictly prohibited.**

---

This repository contains a Proof of Concept of the Front-prop algorithm.

See the [`frontprop_fast.ipynb`](https://github.com/rohrl/frontprop/blob/main/frontprop_fast.ipynb) notebook for a quick demo.

### Summary

Front-prop is a proposal of an unsupervised representation learning algorithm which tries to address some of the limitations of backpropagation (see below).

Learning is done through inference and is unbound. It does not require learning objective, it's aim is to fit to patterns in the incoming data and learn efficient representations of these patterns (a.k.a. embeddings). Therefore it should be able to adapt to distribution shifts in real data.

Note that in most cases Front-prop can be adopted to modern Neural Network architectures (e.g. CNNs), by just replacing backpropagation, and therefore it can leverage their well-researched advantages and achievements.

### Motivation

The motivation behind Front-prop is to seek alternative learning algorithms to backpropagation, which are more similar to how humans and other intelligent beings learn. The goal is to find ones that are better suited for achieving AGI, than backpropagation.

Backpropagation is wonderful learning algorithm and has been the foundation of the latest breakthroughs in AI. But it suffers from several constraints that make it unlikely to be the way how real, biological brains work and probably insufficient for AGI  ([LeCun, 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf); [Hinton, 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf)). 

Some of the limitations of backpropagation are: 
1. complete separation of training phase and inference phase (sepearation in time, in data, and in how each operates),
1. requirement for explicitely provided expected outputs, 
1. inability to continuously learn and infer at the same time
1. learning is bounded (training data is defined upfront and training has to end before inference can start)
1. distribution shifts between real data and training data
1. requires saving each layer's forward results to be used for the backward pass
1. computationally intensive 
1. requires knowledge of derivatives of each layer

(this list is not exhaustive)


Front-prop algorithm is partially inspired by the Hebbian theory ([Hebb, 1949](https://en.wikipedia.org/wiki/Organization_of_Behavior)) of how neurons in the brain operate and by the Perceptron model ([McCulloch, Pitts, 1943](https://www.bibsonomy.org/bibtex/13e8e0d06f376f3eb95af89d5a2f15957/schaul)).


### Detailed description

TODO.
(for now see implementation or contact the author)


---

Copyright © 2021 **Karol Pajak**
