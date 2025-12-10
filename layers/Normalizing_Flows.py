import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows.base import Flow
from nflows.transforms import CompositeTransform

from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.nn.nets import ResidualNet
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.normalization import BatchNorm, ActNorm

def create_conditional_nsf_flow(configs):
    feature_dim, context_dim, num_layers, hidden_features, num_bins = (
        configs.enc_in * configs.patch_len,
        configs.d_model,
        configs.flow_layers,
        configs.hidden_features,
        configs.num_bins
    )

    transforms = []
    mask = torch.zeros(feature_dim)
    mask[:feature_dim // 2] = 1

    for i in range(num_layers):
        transforms.append(RandomPermutation(feature_dim))
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask = mask if i % 2 == 0 else 1 - mask,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_blocks=2,
                    activation=F.leaky_relu,
                ),
                num_bins=num_bins, 
                tails="linear", 
                tail_bound=5.0, 
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
            )
        )
        transforms.append(BatchNorm(features=feature_dim))
        
    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([feature_dim])
    )

