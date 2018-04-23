import torch.nn as nn
import torch.nn.functional as F

class DownSampleH(nn.Module):
    def __init__(self, config):
        super(DownSampleH, self).__init__()

        pool_kernel = 3
        pool_stride = 2
        pool_num = 2
        pool = [ nn.MaxPool1d(pool_kernel, stride=pool_stride) for i in range(pool_num) ]

        def pool_dim_func(k, size):
            if k < 2:
                return (size - (pool_kernel - 1) - 1) // pool_stride + 1
            return (pool_dim_func(k-1, size) - (pool_kernel - 1) - 1) // pool_stride + 1


        class Flatten(nn.Module):
            def __init__(self):
                super(Flatten, self).__init__()

            def forward(self, x):
                s = x.size()
                return x.view(s[0],-1)

        linear_feature = pool_dim_func(pool_num, config.h_size)* config.embed_size* config.bi
        linear = [ Flatten(), nn.Linear(linear_feature, 1) ]

        self.downsample = nn.Sequential(*(pool+linear))

    def forward(self, h):
        return self.downsample(h)
