## Author's raw notes

### Ideas
* if a neuron strays to local optima (gets barely any activations):
  * randomise weights gradually, or
  * "shock therapy" - re-initialise with random weights
* feed label as input for training (maybe mask sometimes?)
* try variable hyperparams eg each neuron getting the value of `threshold_decay` and `weight_boost` from normal distro 


### Observations

* often many neurons converge to same pattern - use some repelling mechanism to push neurons within same layer away from converging on same pattern

* VERY GOOD NEWS:
  * it ALWAYS converges and is STABLE afterwards, with all hyperparams remaining SAME througout the whole process
  * probability distribution of different patterns does not seem to affect it much - ie even when pattern's probabilties are not same (eg [0.43 0.43 0.13]) there is no bias towards more neurons converging on most frequent patterns
  
* some numbers: for 3 patterns 3x3, after about 100 iterations it's already stable; for 3 patterns 4x4, about 200 is enough

* MNIST: Tried 20 neurons in single layer (10k-50k iters), but they all converge to some avergaged blob. 
  * Also tried feeding in the label as extra row - this doesn't seem to ever be picked up in weights (but does it make a difference?? maybe it's not ignored and influences how neurons specialise hmm)
  * Conclusion: single layer is too shallow for this - obviously a linear function (=single layer) can't model such large input space (28x28), we need HIERARCHY (more than 1 layer). Convolutions would be ideal. But maybe just hierarchy will be enough - see `mnist` notebook - when all samples per class are squashed onto single image, the digit is still clearly visible - so we don't need the network to be translation invariant, hence convolutions are maybe not necessary for this task.
  * --> Try building a hierarchy and also try feeding labels (at the top of hierarchy, not bottom)
  * Also: after peaking at around 20k iters, weights start to fade and decay to zeros (mostly) - how to prevent it? :(
  * when tried lot more neurons (200) - I could see some specialisation eg could distinguish 0s, 5s, 9s, 8s, 6s in some. I never could find 1s, 4s, 7s though. Interesting...
  
* there needs to be lot more neurons than classes, so that random init can explore the space sufficiently to descent to different specialisations. With more classes the chance of having a neuron specialise for each class decreases (exponentially?). Hierarchy should fix this - adding more layers should reduce the number of classes per layer (ie low layers detecting primitive local patterns, higher layers patterns on top of patterns etc)

* **????** WHY when I used cosine similarity and boosting via vector math (moving W vector towards input vector in the feature space) - this didn't work at all - all neurons converged on a same pattern that was some weird average. Tried different hyperparams - didn't help 
  * actually, turns out this happens even when ONLY cosine similarity is used, and also when ONLY vector geometry for boosting is used 
  * actually2, turns out Cosine similarity ignores vector's length so it's obviously not great for us. MSE, which is more like Euclidian distance, works heaps better.
  
* image reconstruction results are good - it picks up edges well, but bit blurry. However, randomly initialised kernels (both uniform and Gaussian) also reconstruct image very well (better actually - less blurry)  
  
* tried Gaussian initialization: 
  * 3 patterns: TODO
  * MNIST: crap, all converge to some diagaonal line - common trait of 1, 7 and 9 I guess
  * Cats & dogs: less variety in kernels, it seems
  
* added noise to the simple patterns experiment - it still learns them, but higher noise rate exponentially slows down learning, and if more than 10% samples are noise, it does not converge.
  * also observed: at noise rate of 5%, most neurons converge. But some fail to converge to any of the patterns - but they all converge to the same value (!), and then all of them seem to go in circles around the feature space, following the trail of last inputs and moving in same direction. Interesting...
    * (the algorithm for killing similar neurons would fix this)
    
  * conclusion --> input should mostly consist of repeatable patterns, total noise is very unwelcome. So we should limit the amount of possible patterns in feature space by keeping input space small and tackling complexity by adding layers.

### TODO
* see if new patterns are shown, if it can unlearn old and learn new
* ~feed MNIST~
  * ~IS MNIST BALANCED ??? cound samples per class!~ it's pretty balanced
  * ~try on fewer and more distinctive classes~ didn't help
  * it seems to always converge just on one class - I think I know where the problem might be - the more pixels are active in a pattern, the higher signal for the activation - so classes with most pixels will always win (?) - verify this on the simple patterns dataset - solution: need to re-balance that by some scaling
* try hierarchy (more layers)
* try convolutions, like AlexNet
* measure impact of class prob distribution on the chance of neuron specialising for it
* fix outputs (stretch to 0-1 ?)
* try out setting initial activation threshold at levels other than 1.0 (maybe 0?)
* when more than 1 layer: skip inputs that are all zeros (waste of time), but initialise the threshold lower (see above)
* try [-1,1], or unbound ?


### Alex feedback
* try reconstructing using random kernels, for comparison (use Gaussian noise)
* have a look at Boltzman Machines

### Ash's observations
* X (TODO how many?) patterns, Y iterations - learnt...

		
## Other ideas	

* take any CNN and train it to reduce the entropy of output (or trailing avg entropy, or entropy of weights or sth like that) - to create a "generic compressor" of data, without labels