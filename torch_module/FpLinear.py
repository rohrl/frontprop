import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: normalise input and weights + use gaussian initialisation

T_DECAY_DEFAULT = 0.0005
W_BOOST_DEFAULT = 0.02

class FpLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, t_decay=T_DECAY_DEFAULT, w_boost=W_BOOST_DEFAULT, device=None, dtype=None):
        if bias:
            raise Exception("Bias not supported for FrontPropLinear")
        
        super(FpLinear, self).__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

        # learning is through front propagation only
        self.weight.requires_grad = False

        self.device = device

        # init using Gaussian
        nn.init.normal_(self.weight, mean=0, std=1)
        # and normalise
        self.weight.data = self.__normalise_unitary(self.weight)

        # hyper params
        self.t_decay = t_decay
        self.w_boost = w_boost

        self.frozen = False

        assert self.t_decay > 0        
        assert self.w_boost > 0

        # init thresholds
        self.t = torch.ones(out_features, device=device, dtype=self.weight.dtype)
    

    def __normalise_unitary(self, data, dim=1):
        return data / torch.norm(data, dim=dim, keepdim=True)
    

    def __get_weights_boost(self, data_vector):
        assert data_vector.shape == self.weight.shape

        w_boost = self.w_boost * (data_vector - self.weight)
        excited_filter = self.excitations.unsqueeze(1).expand_as(self.weight)
        w_boost = w_boost * excited_filter

        assert w_boost.shape == self.weight.shape
        return w_boost
    

    def forward_single_sample(self, data):
        
        data_vector = data.expand(self.out_features, -1)
        data_vector = self.__normalise_unitary(data_vector)

        # dot product of input and weights
        # (equivalent to cos_sim because both are unit vectors)
        output = torch.sum(data_vector * self.weight, dim=1)

        assert torch.all(output > -1.01) and torch.all(output < 1.01)

        self.excitations = (output >= self.t).float()

        if not self.frozen:
            self.weight.data = self.weight + self.__get_weights_boost(data_vector)
            self.weight.data = self.__normalise_unitary(self.weight)
            self.t = self.excitations * output + (1.0 - self.excitations) * self.t
            self.t = self.t * (1.0 - self.t_decay)
       
        self.__assert()
        
        return output
    

    def forward(self, input):
        # FIXME:
        # As of now, samples are just processed sequentially, for simplicity.

        assert input.shape == (input.shape[0], self.in_features)

        self.output = torch.zeros(input.shape[0], self.out_features, device=self.device, dtype=self.weight.dtype)

        for i, sample in enumerate(input):
            sample_out = self.forward_single_sample(sample)
            self.output[i] = sample_out

        assert self.output.shape == (input.shape[0], self.out_features)
    
        # ReLU-like non-linear transformation via cutoff threshold
        #
        #   TODO: Should we output the absolute value or only the diff above threshold ?
        #   (note the threshold changes on each pass)
        self.output = torch.where(self.output >= self.t, self.output, torch.zeros_like(self.output))

        # project outputs onto a sphere, as expected by downstream layer
        self.output = self.__normalise_unitary(self.output, dim=1)

        return self.output

    def backward(self, grad_output):
        raise Exception("Backward pass not implemented for FrontPropConv2d")


    def freeze(self):
        self.frozen = True
            

    def unfreeze(self):
        self.frozen = False


    def __assert(self):
        assert self.weight.shape == (self.out_features, self.in_features)
        assert self.t.shape == (self.out_features,)

        assert torch.allclose(torch.norm(self.weight, dim=1), torch.ones(self.out_features, device=self.device), atol=1e-5)
        assert torch.all(self.t > 0)

        assert self.excitations.shape == (self.out_features,)
