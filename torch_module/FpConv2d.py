import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

# TODO: normalise input and weights + use gaussian initialisation

T_DECAY_DEFAULT = 0.0005
W_BOOST_DEFAULT = 0.02


class FpConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', t_decay=T_DECAY_DEFAULT, w_boost=W_BOOST_DEFAULT, device=None, dtype=None):
        if bias:
            raise Exception("Bias not supported for FrontPropConv2d")
        if dilation != 1:
            raise Exception("Dilation not supported for FrontPropConv2d")
        if groups != 1:
            raise Exception("Groups not supported for FrontPropConv2d")

        super(FpConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                       padding_mode, device, dtype)

        # learning is through front propagation only
        self.weight.requires_grad = False

        # init using Gaussian
        nn.init.normal_(self.weight, mean=0, std=1)

        # TODO: add non-square support
        if self.kernel_size[0] != self.kernel_size[1]:
            raise Exception("Non-square kernel not supported for FrontPropConv2d")
        if self.stride[0] != self.stride[1]:
            raise Exception("Non-square stride not supported for FrontPropConv2d")
        if self.padding[0] != self.padding[1]:
            raise Exception("Non-square padding not supported for FrontPropConv2d")

        self.kernel_size = self.kernel_size[0]
        self.stride = self.stride[0]
        self.padding = self.padding[0]

        self.device = device

        # hyper params
        self.t_decay = t_decay
        self.w_boost = w_boost

        self.frozen = False

        assert self.t_decay > 0
        assert self.w_boost > 0

        # see lazy_init_thresholds()
        self.t = None

    def lazy_init_thresholds(self, out_h, out_w):
        if self.t is None:
            self.t = torch.ones(self.out_channels, out_h, out_w, device=self.device, dtype=self.weight.dtype)

    def __normalise_unitary(self, data, dim):
        return data / torch.norm(data, dim=dim, keepdim=True)

    def __get_weights_boost(self, kernel_idx, data_tensor):
        # FIXME: can i use this addition, or do i need to use the angle?

        kernel_weights = self.weight[kernel_idx]

        assert data_tensor.shape == kernel_weights.shape

        w_boost = self.w_boost * (data_tensor - kernel_weights)

        return w_boost

    def __get_input_patch(self, sample, out_h, out_w):
        in_h = out_h * self.stride - self.padding
        in_w = out_w * self.stride - self.padding
        return sample[:, in_h: in_h + self.kernel_size, in_w: in_w + self.kernel_size]

    def forward(self, mini_batch):

        # TODO: currently mini batch is done sequentially

        outputs = []

        for sample in mini_batch:
            out = self.forward_single(sample)
            outputs.append(out)

        return torch.stack(outputs)

    def forward_single(self, sample):

        assert sample.shape[0] == self.in_channels

        output = F.conv2d(sample.unsqueeze(0), self.weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups).squeeze()

        self.lazy_init_thresholds(output.shape[-2], output.shape[-1])

        assert self.t.shape == output.shape

        # ReLU-like non-linear transformation via cutoff threshold
        output = torch.where(output >= self.t, output, torch.zeros_like(output))

        # Learning happens below:
        #
        #   For each location where the threshold was exceeded, update the weights.
        #   The weights are updated by a small amount towards the input data tensor
        #   and thresholds are set to the new value where the threshold was exceeded.
        #   Thresholds are also decayed by a small amount on each pass.
        #
        #   Convolution kernels' weights are updated in random order of locations.
        #
        #   The output is equal to the activation above the threshold (or zero if below threshold).
        #
        #   ---
        #   TODO: Should we output the absolute value or only the diff above threshold ?
        #   (note the threshold changes on each pass)
        #
        #   FIXME: all inputs should be first normalised - this is tricky
        #   TODO: Optimise this
        #   * Try removing loops
        #   * maybe use torch.sparse_coo_tensor() to save memory
        if not self.frozen:

            # Update weights:

            excitations_idxs = torch.nonzero(output)
            # shuffle indices randomly
            excitations_idxs = excitations_idxs[torch.randperm(excitations_idxs.shape[0])]
            # iterate locations where threshold exceeded and shift weights closer to the input
            for kernel_idx, h, w in excitations_idxs:
                data_tensor = self.__get_input_patch(sample, h, w)
                assert data_tensor.shape == (self.in_channels, self.kernel_size, self.kernel_size)
                data_tensor = self.__normalise_unitary(data_tensor, dim=(1, 2))
                assert data_tensor.shape == (self.in_channels, self.kernel_size, self.kernel_size)
                # update weights
                self.weight[kernel_idx] += self.__get_weights_boost(kernel_idx, data_tensor)
                # normalise weights
                self.weight[kernel_idx] = self.__normalise_unitary(self.weight[kernel_idx], dim=(1, 2))
                assert self.weight[kernel_idx].shape == (self.in_channels, self.kernel_size, self.kernel_size)

            # Update thresholds:

            binary_excitations = (output > 0)
            # where threshold exceeded set it to the new value
            self.t = binary_excitations * output + ~binary_excitations * self.t
            # decay all thresholds
            self.t = self.t * (1.0 - self.t_decay)

        return output

    def backward(self, grad_output):
        raise Exception("Backward pass not implemented for FrontPropConv2d")

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
