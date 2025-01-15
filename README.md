# Frontprop

_Continual, unsupervised learning in forward pass._

## Background

The idea of Frontprop started from curiosity of what can be achieved with a simple implementation of [Hebbian-based](https://en.wikipedia.org/wiki/Hebbian_theory) learning,
fully unsupervised, and how to make it more compatible with current AI algorithms and architectures, with the goal of searching for alternatives to backpropagation (see [Motivation](#motivation)).

Frontprop is an unsupervised learning algorithm - it does not try to optimise for specific expected output or a specific task, bur rather aims to produce useful representations of inputs, regardless of the domain. 
Useful in the sense of capturing common patterns in the input data. 
As such there is no explicit training objective and learning is achieved by continuously fitting to patterns in the incoming data and producing compressed (fewer-dimensional) representations of these patterns ([embeddings](https://developers.google.com/machine-learning/crash-course/embeddings)).
This process can be viewed as a way of (lossy) compression of the input data, which is what intelligence is thought to be (see [Ilya's talk](https://www.youtube.com/live/AKMuA_TVz3A?si=YgpcGdQPSE0VGdZg)).

Such embeddings can then also be used as input for smaller adapter networks, trained in supervised manner for more specific, downstream tasks, for which expected outputs are more easily available.

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

Frontprop is a modification of the original [Hebb's rule](https://en.wikipedia.org/wiki/Hebbian_theory) (which is the commonly held belief on how neurons in our brains learn), applied to the artifical neuron - [Perceptron](https://en.wikipedia.org/wiki/Perceptron).

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
The layer's output is a **vector** of all individual neurons outputs.

$\mathbf{y} = \phi(\mathbf{W} \mathbf{x} + \mathbf{b})$

Such layer design is used in a Linear layer (a.k.a. "fully connected" or "feed forward") in AI models 
(e.g. [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)), and also in a Frontprop linear layer [`FpLinear`](./torch_module/FpLinear.py).
Currently Frontprop also implements a convolutional layer [`FpConv2d`](./torch_module/FpConv2d.py), similar to `torch.nn.Conv2D`.


#### Learning algorithm and weight update rule

[*the logic described in this section is subject to ongoing experimentation and may change*]

Unlike in global gradient backpropagation, learning in Frontprop is localised to a neuron, simply:

```
if (w * x) > excitation_threshold:
  update_neuron_weights()
  excitation_threshold = w * x
  output = w * x
else:
  excitation_threshold *= (1 - threshold_decay)
  output = 0
```

`x * w` is simply a dot product of input and neuron's weights.

`excitation_threshold` is a scalar, part of internal neuron state, alongside with its weights.

If neuron was activated (excited) with the input (i.e. the product exceeded the activation threshold),
the weights are updated (see the update rule [below](#update-rule)) and the product is the output
(the "job" of activation function is implicitly realised by the excitation threshold, explained [below](#activation-function).

If the neuron was not activated, the activation threshold is lowered by a small amount, and the output is `0`.

Neurons weights are randomly initialized at the beginning of the training.

#### Update rule

Weights are updated following the Hebb's principle, i.e. reinforce the connections that contributed to the activation.
This can be implemented in a multitude of ways, but I found the following simple logic to work well:

$\Delta\mathbf{w} = \lambda(\mathbf{x} - \mathbf{w})$

$\mathbf{w} := \mathbf{w} + \Delta\mathbf{w}$

where the change is simply a vector subtraction (in high dimensional space), 
scaled by the learning rate $\lambda$ (which in code is called `weight_boost`).

![](https://upload.wikimedia.org/wikipedia/commons/2/24/Vector_subtraction.svg)

In other words, we are moving the weight vector **slightly** closer to the input vector,
which will result in neuron being "more sensitive" to this input,
increasing the chance of activations from this or similar inputs.

Our simple update rule is in fact similar to [Oja's rule](https://en.wikipedia.org/wiki/Oja%27s_rule):

$\Delta\mathbf{w} = \lambda y (\mathbf{x} - y \mathbf{w})$

except in Oja's rule:
1. weights are always updated and the update "strength" is proportional to the activation (1st $y$), 
while in Frontprop it's discrete: no update if activation is below the threshold,
2. the update vector is additionally "boosted" by the activation "strength" (2nd $y$)

Lastly, we set the `excitation_threshold` to be at the level of the output, which caused the neuron to fire.

#### Intuition

The intuition is that we can think of inputs and outputs, and weights,
as points in the high-dimensional embedding space, representing some abstract meaning.

At the beginning there is hardly any activations, so neurons keep lowering their thresholds.

Imagine the neuron's weight vector as a point in this space, and activation threshold being
a radius of a hypersphere centered in that point.

The hypersphere radius keeps growing (i.e. threshold lowering) until it spans a region of input space
where signals happen to be present. This will cause the neuron to fire, and so trigger
the weight update, which will effectively move the hypersphere slightly closer to the input.

Eventually, neurons will specialise in firing for repeating/dominant patterns in inputs,
and the random initialisation provides some initial coverage of the space to "spread" the search.

If input distribution changes and new patterns emerge, neurons will adapt again by "growing their hypersphere radius", 
if not excited.

This process is somewhat similar to clustering algorithms, like KMeans, but expressed as
neuronal behaviour (Hebb's rule).

One drawback of the current logic is that multiple neurons may converge in the same spot,
but implementing some form of [Lateral Inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition) (which reportedly also takes place in the brain),
for example via some "repelling force" or total energy minimisation, should prevent it
(early "brute-force" implementation of this have shown that it does improve performance).

Another weakness is that rare patterns will not attract neurons as much as frequent patterns
(though this does not seem to be a strong effect, see below in [Observations](#observations)).
The above Lateral Inhibition should help, but perhaps optimising some entropy-inspired layer-wise energy
could be another solution 
(as in rare patterns carrying more information, which can be measured using [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)))
(<- TODO).

#### On activation function and bias

In contrast to the original Perceptron, or a typical layer in ML models,
Frontprop neurons don't use an activation function and bias.

Non-linear activation function is critical in ML models, otherwise the entire computation
would reduce to a linear transformation (each layer performing one, which stacked together is still a linear transformation).

The "job" of non-linear activation function is performed by the activation threshold in Frontprop.
Its non linear behaviour (zeroing the output when neuron did not fire) is very similar to the popular 
[`ReLU`](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function used broadly in ML models.

When it comes to the lack of *bias* in Frontprop layers, it has been evidenced by well performing models, like Llama,
that bias is not necessary.

#### Hyperparameters

There are 2 global hyperparameters that control learning process speed and the level of specialisation of neurons:
1. `weight_boost` (i.e. the learning rate used in weight update)
2. `threshold_decay` which is the neurons' thresholds drop off rate (when not activated)

#### Normalization

To avoid the problem of weights reducing to zero or exploding to infinity, inputs are expected to be
normalised to unitary vectors. Same normalisation is applied to weights and outputs.
This may not be necessary though, as the algorithm used to work without it.

## Observations

Frontprop is still very much work in progress, but some early, promising observations have been made.

TODO: converging on simple patterns (+stability)

TODO: pattern probability and noise impact

TODO: comparison to unsupervised (log probe, knn)

TODO: plot neurons count vs score on MNIST

TOOD: conv layer detecting patterns

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
Backpropagation not only requires a lot of compute resources, but due to the global gradient propagation they are highly centralised,
with high bandwidth connections, resulting in very expensive datacenters with mind-blowing energy and cooling requirements.
While there is a growing research on decentralised training, it is still in its infancy and difficult to make it work with backpropagation.

Training techniques that use only local learning, like Frontprop, if proven to work, could facilitate massively distributed and decentralised training,
which is much more efficient, cheaper, more sustainable and eco-friendly, and will unlock next level scale of compute
(see Pluralis Research [in-depth article](https://www.pluralisresearch.com/i/146390174/myth-the-swarm-can-never-get-big-enough) on the case for decentralised learning])

---

## Other backpropagation alternatives

| Method                                                                                                                                                    | Supervised | MNIST acc    | similarities | differences |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------|--------------|-------------|
| [Forward-Forward](https://arxiv.org/abs/2212.13345)                                                                                                       | no         | 98-99% (MLP) |              |             |
| [Signal Propagation](https://arxiv.org/abs/2204.01723)                                                                                                    | yes        |              |              |             |
| [Oja's Rule](https://en.wikipedia.org/wiki/Oja%27s_rule) and [generalized Hebbian algorithm](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm) | no         |              |              |             |
| [Boltzman machines](https://en.wikipedia.org/wiki/Boltzmann_machine)                                                                                      | both       |              |              |             | 
| [Generalised Hebbian Algorithm (Sanger's Rule)](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm)                                              | no         |              |              |             |
| [Equilibrium Propagation](https://arxiv.org/abs/1602.05179)                                                                                               | yes        |              |              |             |
| Spiking networks                                                                                                                                          |            |              |              |             | 

---

## Limitations

1. At the moment, additional layers do not improve results, when training from scratch (WIP)
2. Bias is not supported.
3. Needs some form of [Lateral Inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition) to prevent more than one neuron converging on the same pattern, so that neurons are utilised more effectively.
4. Current implementation lacks inhibition mechanism, which is known to exist in the brain and in AI models (negative weights). 

---

## Citation

If you find this work useful in your research, please consider citing:

```
@software{frontprop,
  author = {Pajak, K.},
  title = {Frontprop},
  url = {https://github.com/rohrl/frontprop},
  year = {2024}
}
```
