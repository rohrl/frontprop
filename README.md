# Frontprop

_Continual, unsupervised learning in forward pass._

## Background

The idea of Frontprop started from curiosity of what can be achieved with a simple implementation of [Hebbian-based](https://en.wikipedia.org/wiki/Hebbian_theory) learning,
fully unsupervised, and how to make it more compatible with current AI algorithms and architectures, with the goal of searching for alternatives to backpropagation (see [Motivation](#motivation)).

Frontprop is an unsupervised learning algorithm - it does not try to optimise for specific expected output or a specific task, bur rather aims to produce useful representations of inputs, regardless of the domain. 
Useful in the sense of capturing common patterns in the input data. 
As such there is no explicit training objective and learning is achieved by continuously fitting to patterns in the incoming data and producing compressed (fewer-dimensional) representations of these patterns ([embeddings](https://developers.google.com/machine-learning/crash-course/embeddings)).
This process can can be viewed as a way of (lossy) compression of the input data, which is what intelligence is thought to be (see [Ilya's talk](https://www.youtube.com/live/AKMuA_TVz3A?si=YgpcGdQPSE0VGdZg)).

Such embeddings can then also be used as input for smaller adapter networks, trained in supervised manner for more specific, downstream tasks, for which expected outputs are easily available.

In Frontprop, there is no separation between training and inference - it only uses forward pass (as opposed to "backprop") and each pass changes the network, to continuously adapt it to incoming data - similar to how organic brains are commonly assumed to work. 
Consequently, there is no stop condition for training, and therefore the algorithm natively implements Continual Learning and should be able to naturally handle distribution shifts **[unverified]**, which are present in real world environments.

In most cases Frontprop should be easy to adapt to modern Neural Network architectures (e.g. CNNs, Transformers), just by replacing backpropagation, without modifications to the model architecture.
Current implementation includes a [`FpLinear`](./torch_module/FpLinear.py) and [`FpConv2d`](./torch_module/FpConv2d.py) layers, implemented as `torch` modules. 
They can be used to replace its corresponding backpropagation layers (see [Limitations](#limitations)), for easy experimentation with existing architectures.

Frontprop should also be less compute/memory demanding **[unverified]**, as it does not have to store activations, nor perform the backward pass.

---

## How it works

TODO: demo notebook
 [`frontprop_fast.ipynb`](https://github.com/rohrl/frontprop/blob/main/frontprop_fast.ipynb) notebook for a quick demo.

Frontprop is a modification of the original Hebb's rule (which is the commonly held belief on how neurons in our brains learn), applied to the artifical neuron - [Perceptron](https://en.wikipedia.org/wiki/Perceptron).

Hebb's rule states that learning process in neurons is a consequence of reinforcing connections (synapses) that contributed to neuron firing (excitation).

However, a naive implementation of Hebb's rule, i.e. simply reinforcing (multiplying) the weights that contributed to neuron's activation, is unstable as the weights will tend to explode to infinity.

Another challenge is managing/guiding neuron's excitations in the absence of explicit expected output. 
Backpropagation is inherently supervised - even in weakly or self supervised settings the expected output needs to be provided in order to compute weight updates via gradient descent.

Frontprop, on the other hand, is fully unsupervised, and learning happens entirely from inputs and their flow through the network.
Since there are no expected outputs, there's also no expectation on when a neuron should fire (as opposed to supervised learning where output neurons are expected to "fire" according to provided training labels).

Turns out that just by adapting to inputs, Hebb-inspired learning algorithms, such as [Oja's rule](https://en.wikipedia.org/wiki/Oja%27s_rule) can learn useful things -
like computing [Principal Components](https://en.wikipedia.org/wiki/Principal_component_analysis).

### Frontprop neuron

In a [Perceptron](https://en.wikipedia.org/wiki/Perceptron) (artificial neuron), input vector `x` is multiplied by weights `w` (dot product), 
then the result (scalar) is added to bias `b`, which is then fed into activation function $\phi$ (`phi`) to produce the scalar neuron output `y`, as in the formula below:

$y = \phi(\mathbf{w} \cdot \mathbf{x} + b)$

A layer is a collection of such neurons, which receive same input `x`, and so the layer is represented by a weight **matrix** `W` - a stack of all layer's neurons weights, bias **vector** `b` and a shared activation function.
The layer output is a **vector** of individual neurons outputs.

$\mathbf{y} = \phi(\mathbf{W} \mathbf{x} + \mathbf{b})$

Such layer design is used in a Linear layer (a.k.a. "fully connected" or "feed forward") in AI models 
(e.g. [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)), and also in a Frontprop linear layer [`FpLinear`](./torch_module/FpLinear.py).
Currently Frontprop also implements a convolutional layer [`FpConv2d`](./torch_module/FpConv2d.py), similar to `torch.nn.Conv2D`.


#### Learning algorithm and weight update rule

TODO 

```
if.. then
else ..
```

$formula$

most similar to Oja's rule, which is known to find principal components of the input. 

Intuition:

#### Normalization

...

#### Hyperparameters

...

---

## Observations

Frontprop is still very much work in progress, but some early, promising observations have been made.

TODO: plot neurons count vs score on MNIST

---

## Roadmap

### Done

- [x] verify convergence (empirically, no proof)
- [x] compare single layer Frontprop to baselines:
  - [x] supervised
    - [x] LogReg
    - [ ] CNN
  - [x] unsupervised
    - [x] KMeans
      - [x] with trained LogReg probe
      - [x] with KNN
    - [x] Forward-Forward
    - [ ] PCA
  - [x] randomly initialized, untrained 
- [x] research existing *forward learning* methods
- [x] HP grid search
- [x] fast implementation of *Frontprop* `Linear` and `Conv2D` layers in `torch` (see `torch_module/`)

### TODO / In progress

- [ ] **use Frontprop to fine-tune a pretrained language model**
- [ ] use Frontprop to train/fine-tune a CNN
- [ ] understand why multi-layer FF does not improve over single layer 
- [ ] successfully train a deep model (e.g. CNN)
- [ ] evaluate in **continual learning** setting
- [ ] evaluate in supervised setting: labels mixed into inputs in training
- [ ] evaluate on more datasets (CIFAR, Fashion-MNIST etc.)
- [ ] support parallel mini batches
- [ ] reduce neuron redundancy (neurons with similar weights) by sth like [Lateral Inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition)
- [ ] formal proof of convergence
- [ ] try [Oja's Rule](https://en.wikipedia.org/wiki/Oja%27s_rule) weight update formula
- [ ] compare Frontprop-trained weights with backprop-trained weights (see MI libraries)
- [ ] come up with a better name :)

---

## Motivation

#### Biological inspiration

The motivation for the research behind Frontprop is to seek alternative learning algorithms to backpropagation, which are more similar to how biological brains learn.

My personal intuition, from watching how babies learn, is that it's fully unsupervised, where they just observe the world, 
trying to pick up on patterns and correlations, which later can be used to make predictions about their environment.
Only later, once this foundational "pretraining" is complete, we use "labels" (expected outputs) to augment our learning (i.e. supervised learning).

Backpropagation has been the foundation of Deep Learning and facilitated the amazing breakthroughs in AI. Nevertheless, it does have some limitations:  

1. strict separation between training phase and inference phase (separation in time, in data, and in kinds of operations),
1. necessity to explicitly provide expected outputs and/or a task/domain-specific loss function 
1. inability to continuously learn and infer at the same time
1. necessity to obtain all training data upfront, and before inference phase
1. requires saving each layer's forward results (activations) to be used for the backward pass
1. requires knowledge of derivatives of each layer
1. sequential computation (layer by layer)

Many of the above are hard to reconcile with our observations of how biological brains learn in nature. 
Perhaps looking beyond just backpropagation, and drawing more knowledge from nature, could help us on the path to more general artificial intelligence 
([LeCun, 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf); [Hinton, 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf)).

#### Efficiency

Furthermore, current training techniques of large models are insanely energy demanding and any efficiency optimisations there would be very welcome.
Backpropagation not only requires a lot of compute resources, but due to the global gradient propagation they need to be highly centralised,
with high bandwidth connections, resulting in very expensive datacenters with mind-blowing energy and cooling requirements.

Training techniques that use only local learning, like Frontprop, if proven to work, could facilitate massively distributed and decentralised training,
which is much more efficient, cheaper, more sustainable and eco-friendly, and will unlock next level scale of compute
(see Pluralis Research [in-depth article](https://www.pluralisresearch.com/i/146390174/myth-the-swarm-can-never-get-big-enough) on the case for decentralised learning])

---

## Other backpropagation alternatives


| Method                                                                                                                                                     | Supervised | MNIST acc    | similarities | differences |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------|--------------|-------------|
| [Forward-Forward](https://arxiv.org/abs/2212.13345)                                                                                                        | no         | 98-99% (MLP) |              |             |
| [Signal Propagation](https://arxiv.org/abs/2204.01723)                                                                                                     | yes        |              |              |             |
| [Oja's Rule](https://en.wikipedia.org/wiki/Oja%27s_rule) and [generalized Hebbian algorithm](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm)  | no         |              |              |             |
| [Boltzman machines](https://en.wikipedia.org/wiki/Boltzmann_machine)                                                                                       | both       |              |              |             | 
| [Generalised Hebbian Algorithm (Sanger's Rule)](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm)                                               | no         |              |              |             |
| [Equilibrium Propagation](https://arxiv.org/abs/1602.05179)                                                                                                | yes        |              |              |             |
| Spiking networks                                                                                                                                           |            |              |              |             | 


---

## Limitations

1. At the moment, additional layers do not improve results, when training from scratch (WIP)
2. Bias is not supported.
3. Needs some form of [Lateral Inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition) to prevent more than one neuron converging on the same pattern, so that neurons are utilised more effectively.
4. Current implementation lacks inhibition mechanism, which is known to exist in the brain and in AI models (negative weights). 

---

## Citation

If you find this repository useful in your research, please consider citing:

```
@software{frontprop,
  author = {Pajak, K.},
  title = {Frontprop},
  url = {https://github.com/rohrl/frontprop},
  year = {2024}
}
```
