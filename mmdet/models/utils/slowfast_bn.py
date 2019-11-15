from __future__ import division

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init


class _SlowFastBatchNorm(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     # 'slow_weight', 'slow_bias', 
                     'slow_running_mean', 'slow_running_var',
                     'num_features', 'affine', 'steps', 'alpha']

    def __init__(self, num_features, steps=4, alpha=0.5, eps=1e-5, 
                 momentum=0.1, affine=True, track_running_stats=True):
        super(_SlowFastBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            #self.slow_weight = Parameter(self.weight.data.clone().detach()) 
            self.bias = Parameter(torch.Tensor(num_features))
            #self.slow_bias = Parameter(self.bias.data.clone().detach())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            #self.register_parameter('slow_weight', None)
            #self.register_parameter('slow_bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('slow_running_mean', torch.zeros(num_features))
            self.register_buffer('slow_running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('slow_running_mean', None)
            self.register_parameter('slow_running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.slow_running_mean.zero_()
            self.slow_running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
            #init.ones_(self.slow_weight)
            #init.zeros_(self.slow_bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def update_slow_fast(self):
        if self.track_running_stats:
            self.slow_running_var = (1 - self.alpha) * self.slow_running_var + self.alpha * self.running_var
            self.slow_running_mean = (1 - self.alpha) * self.slow_running_mean + self.alpha * self.running_mean
            self.running_var.data.copy_(self.slow_running_var)
            self.running_mean.data.copy_(self.slow_running_mean)

        """if self.affine:
            self.slow_weight.data = (1 - self.alpha) * self.slow_weight.data + self.alpha * self.slow_weight.data
            self.slow_bias.data = (1 - self.alpha) * self.slow_bias.data + self.alpha * self.slow_bias.data
            self.weight.data.copy_(self.slow_weight.data)
            self.bias.data.copy_(self.slow_bias.data)
        """

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum 
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # update slow and fast statistics
        if self.num_batches_tracked % self.steps == 0:
            self.update_slow_fast()

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, steps={steps}, alpha={alpha}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_SlowFastBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class SlowFastBatchNorm1d(_SlowFastBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SlowFastBatchNorm2d(_SlowFastBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SlowFastBatchNorm3d(_SlowFastBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))