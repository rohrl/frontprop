import numpy as np
from sklearn.metrics import mean_squared_error

class InvertedNeuron:
    
    float_type = np.float16
    
    def __init__(self, dims, t_decay, w_boost, init=np.random.rand):
        assert len(dims) == 2
        self.dims = dims
        
        # initialize neuron
        self.t = InvertedNeuron.float_type(1.0)
        self.W = init(*dims).flatten().astype(InvertedNeuron.float_type)
        
        self.t_decay = InvertedNeuron.float_type(t_decay)
        self.w_boost = InvertedNeuron.float_type(w_boost)
        
        assert self.t_decay > 0
        assert self.w_boost > 0     
        
    def forward(self, data):
        assert data.shape == np.prod(self.dims)
        
        self.signal = InvertedNeuron.float_type(mean_squared_error(data, self.W))
        
        ''' 
        Logic:
        - if pattern and neuron weights are similar then signal approaches 0
        - if pattern and neuron weights are different then signal approaches 1
        
        Thus, neuron activation/excitement is redefined as if signal <= threshold
        
        If excited: modify weights to be more similar to input pattern and decrease threshold
        If not excited: increase threshold
        '''
        self.excited = self.signal <= self.t
        
        if self.excited:
            self.t = (InvertedNeuron.float_type(1) + self.t_decay) * self.signal # decrease
            self.W += self.w_boost * (data - self.W) # shift weights towards data
        else:
            self.t = (InvertedNeuron.float_type(1) + self.t_decay) * self.t # increase

        return self.signal
    


class OriginalNeuron:
    
    def __init__(self, dims, t_decay, w_boost, init=np.random.rand):
        assert len(dims) == 2
        self.dims = dims
        
        # initialize neuron
        self.t = 1.0
        self.W = init(*dims).flatten()
        
        self.t_decay = t_decay
        self.w_boost = w_boost
        
        assert self.t_decay > 0
        assert self.w_boost > 0        
        
    def forward(self, data):
        assert data.shape == np.prod(self.dims)
        
        self.signal = 1.0 / mean_squared_error(data, self.W)
        self.excited = (self.signal >= self.t)
        
        if self.excited:
            self.t = (1 - self.t_decay) * self.signal
            self.W += self.w_boost * (data - self.W)
        else:
            self.t *= 1 - self.t_decay

        return self.signal
    