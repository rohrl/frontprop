import torch
import torch.nn as nn
import torch.nn.functional as F

T_DECAY_DEFAULT = 0.0005
W_BOOST_DEFAULT = 0.02


class FpConv2d(nn.Conv2d):
    """
    Implements Frontprop 2D convolutional layer, as a torch module.
    Can be used as a direct replacement of torch's nn.Conv2d in CNN architectures.
    At the moment only basic functionality is supported (in/out channels, stride and padding; bias is not supported!).
    """
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

    def __out_for_in(self, in_dim):
        """Calculate convolution output dimension for given input dimension."""
        return (in_dim - self.kernel_size + 2 * self.padding) // self.stride + 1

    def __get_input_patch(self, padded_sample, out_h, out_w):
        # padded_sample.shape = (in_channels, in_h + 2*padding, in_w + 2*padding)
        in_h = out_h * self.stride
        in_w = out_w * self.stride
        return padded_sample[:, in_h: in_h + self.kernel_size, in_w: in_w + self.kernel_size]

    def forward(self, mini_batch):
        # mini_batch.shape = (batch_size, in_channels, in_h, in_w)

        # TODO: currently mini batch is done sequentially

        outputs = []

        for sample in mini_batch:
            out = self.forward_single(sample)
            outputs.append(out)

        return torch.stack(outputs)

    def __get_input_norms(self, sample):
        """
        Sum of dot product with itself gives the length (magnitude) of the vector.
        Compute magnitudes per each location.
        """
        # sample.shape = (in_channels, in_h, in_w)

        single_sample_batch = sample.unsqueeze(0)
        sample_sq = single_sample_batch * single_sample_batch
        # divisor_override=1 turns avg_pool into sum_pool
        sample_sq_sums = F.avg_pool2d(sample_sq,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      divisor_override=1)
        sample_magnitudes = torch.sqrt(sample_sq_sums)
        return sample_magnitudes.squeeze(0)

    def forward_single(self, sample):
        # sample.shape = (in_channels, in_h, in_w)

        assert sample.shape[0] == self.in_channels

        # TODO: redo for-loop into a parallel call
        outputs_per_in_ch = torch.zeros(self.in_channels, self.out_channels,
                                        self.__out_for_in(sample.shape[-2]),
                                        self.__out_for_in(sample.shape[-1]))

        for in_ch_idx in range(self.in_channels):
            outputs_per_in_ch[in_ch_idx] = F.conv2d(sample[in_ch_idx].unsqueeze(0),
                                                    # select the kernel for this input channel
                                                    self.weight[:, in_ch_idx:(in_ch_idx + 1), :, :],
                                                    self.bias, self.stride, self.padding, self.dilation,
                                                    self.groups).squeeze(0)

        assert outputs_per_in_ch.shape == (self.in_channels, self.out_channels,
                                           self.__out_for_in(sample.shape[-2]),
                                           self.__out_for_in(sample.shape[-1]))

        # We wanted input to be normalised too, but it wasn't (because would have to do it per each position separately)
        # Weights were normalised though, so to correct the result we just need to divide it
        # by the lengths (magnitudes) of inputs at each position.
        sample_norms = self.__get_input_norms(sample)
        assert sample_norms.shape == (self.in_channels,
                                      self.__out_for_in(sample.shape[-2]),
                                      self.__out_for_in(sample.shape[-1]))

        # broadcast sample_magnitudes to divide each output channel
        outputs_per_in_ch /= sample_norms.unsqueeze(1)
        output = torch.sum(outputs_per_in_ch, dim=0)
        assert output.shape == (self.out_channels,
                                self.__out_for_in(sample.shape[-2]),
                                self.__out_for_in(sample.shape[-1]))

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
        #   --Convolution kernels' weights are updated in random order of locations.-- (disabled)
        #
        #   The output is equal to the activation above the threshold (or zero if below threshold).
        #
        #   ---
        #   TODO: Should we output the absolute value or only the diff above threshold ?
        #   (note the threshold changes on each pass)
        #
        #   TODO: Optimise this
        #   * Try removing loops
        #   * Maybe just find max activation and update weights only for that one
        #   * maybe use torch.sparse_coo_tensor() to save memory
        if not self.frozen:

            # Update weights:

            excitations_idxs = torch.nonzero(output)

            # --shuffle indices randomly-- (disabled)
            # excitations_idxs = excitations_idxs[torch.randperm(excitations_idxs.shape[0])]

            padded_sample = F.pad(sample, (self.padding, self.padding, self.padding, self.padding))
            assert padded_sample.shape == (self.in_channels,
                                           sample.shape[-2] + 2 * self.padding,
                                           sample.shape[-1] + 2 * self.padding)

            # iterate locations where threshold exceeded and shift weights closer to the input
            for kernel_idx, h, w in excitations_idxs:
                padded_sample_patch = self.__get_input_patch(padded_sample, h, w)
                assert padded_sample_patch.shape == (self.in_channels, self.kernel_size, self.kernel_size)

                # broadcast the sample norms to divide each channel by the norm for this position
                data_tensor_normd = padded_sample_patch / sample_norms[:, h, w].reshape(sample_norms.shape[0], 1, 1)

                assert data_tensor_normd.shape == (self.in_channels, self.kernel_size, self.kernel_size)

                # update weights
                self.weight[kernel_idx] += self.__get_weights_boost(kernel_idx, data_tensor_normd)

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
