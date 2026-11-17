# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Paper target functions evaluated directly on physical input coordinates.

Pass a batch of coordinates; each function returns one column of target values.
Sampling, coordinate scaling, and target standardization belong in the notebook.
"""

import math

import torch


# Figure 4: two-dimensional toy examples, all on [-1, 1]^2.
def toy_polynomial(x):
    """Polynomial toy: 1 + u + v + uv + u^2 - v^2 + 0.5u^2v."""
    u, v = x[:, :1], x[:, 1:2]
    return 1 + u + v + u * v + u**2 - v**2 + 0.5 * u**2 * v


def toy_exponential(x):
    """Smooth toy: exp(0.8u-0.4v) + 0.3 sin(pi*u) cos(pi*v) + 0.2uv."""
    u, v = x[:, :1], x[:, 1:2]
    return (
        torch.exp(0.8 * u - 0.4 * v)
        + 0.3 * torch.sin(math.pi * u) * torch.cos(math.pi * v)
        + 0.2 * u * v
    )


def toy_oscillatory(x):
    """Oscillatory toy: sin(2pi(u+0.5v)) + 0.5cos(pi(1.5u-v)) + 0.15exp(uv)."""
    u, v = x[:, :1], x[:, 1:2]
    return (
        torch.sin(2 * math.pi * (u + 0.5 * v))
        + 0.5 * torch.cos(math.pi * (1.5 * u - v))
        + 0.15 * torch.exp(u * v)
    )


# Figures 5–6: the eight functions in the manuscript's function table.
def gaussian(x):
    """Dimension-normalized Gaussian bump; paper domain [-2, 2]^d."""
    return 10.0 * torch.exp(-2.0 * torch.mean(x**2, dim=1, keepdim=True))


def modified_rosenbrock(x):
    """Modified Rosenbrock with coefficient 2; paper domain [-1, 1]^d, d >= 2."""
    # Keep the paper's coefficient 2 rather than the classical coefficient 100.
    return torch.sum(
        2.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2,
        dim=1,
        keepdim=True,
    )


def modified_zakharov(x):
    """Dimension-normalized Zakharov function; paper domain [-1, 1]^d."""
    dimension = x.shape[1]
    weights = (
        torch.arange(1, dimension + 1, dtype=x.dtype, device=x.device)[None, :]
        / dimension
    )
    # The weighted average is sum(j*x_j)/d^2, keeping its scale comparable across d.
    linear_term = torch.mean(weights * x, dim=1, keepdim=True)
    return torch.mean(x**2, dim=1, keepdim=True) + linear_term**2 + linear_term**4


def log_sum_exp(x):
    """Logarithm of the sum of exponentials; paper domain [-2, 2]^d."""
    return torch.log(torch.sum(torch.exp(x), dim=1, keepdim=True))


def sigmoid(x):
    """Sigmoid of the coordinate average; paper domain [-2, 2]^d."""
    return torch.sigmoid(torch.mean(x, dim=1, keepdim=True))


def griewank(x):
    """Griewank function; paper domain [-2, 2]^d."""
    indices = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)[None, :]
    quadratic_term = torch.sum(x**2, dim=1, keepdim=True) / 4000.0
    cosine_term = torch.prod(torch.cos(x / torch.sqrt(indices)), dim=1, keepdim=True)
    return quadratic_term - cosine_term + 1.0


def rastrigin(x):
    """Rastrigin function; paper domain [-1, 1]^d."""
    return 10.0 * x.shape[1] + torch.sum(
        x**2 - 10.0 * torch.cos(2.0 * math.pi * x), dim=1, keepdim=True
    )


def levy(x):
    """Levy function; benchmark domain [-1, 1]^d; order sweep uses [-5, 5]^d."""
    w = 1.0 + (x - 1.0) / 4.0
    first_term = torch.sin(math.pi * w[:, :1]) ** 2
    middle_term = (w[:, :-1] - 1.0) ** 2 * (
        1.0 + 10.0 * torch.sin(math.pi * w[:, :-1] + 1.0) ** 2
    )
    last_term = (w[:, -1:] - 1.0) ** 2 * (
        1.0 + torch.sin(2.0 * math.pi * w[:, -1:]) ** 2
    )
    return first_term + torch.sum(middle_term, dim=1, keepdim=True) + last_term


# Figures 7–8: uncertainty-quantification targets on the unit hypercube.
def bratley(x):
    """Bratley sum on [0, 1]^d: sum_j (-1)^j prod_{k=1}^j x_k (one-based j)."""
    # Reuse prefix products, with the first term negative.
    signs = x.new_ones(x.shape[1])
    signs[::2] = -1
    return torch.sum(torch.cumprod(x, dim=1) * signs, dim=1, keepdim=True)


def sobol_g_star(x, a, delta, alpha=0.5):
    """Sobol G* on [0, 1]^d with fixed coordinate-wise a and periodic shifts delta.

    The paper uses alpha=0.5, a linearly spaced from 0 to 10, and one fixed
    random shift vector. Construct and save those vectors in the notebook.
    """
    a = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    delta = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
    # Wrap each shifted coordinate into [0, 1) before evaluating its factor.
    shifted = x + delta
    folded = 2.0 * (shifted - torch.floor(shifted)) - 1.0
    factors = ((1.0 + alpha) * torch.abs(folded) ** alpha + a) / (1.0 + a)
    return torch.prod(factors, dim=1, keepdim=True)


# Figure A5: the two adopted nonsmooth examples on [-1, 1]^2.
def abs_corner(x):
    """Absolute-value corner: |u| + |v|."""
    return torch.abs(x[:, :1]) + torch.abs(x[:, 1:2])


def step_ridge(x):
    """Step ridge: +1 for u >= 0 or -1 for u < 0, plus 0.25v."""
    u, v = x[:, :1], x[:, 1:2]
    # Preserve the source convention: the step equals +1 exactly at zero.
    return torch.where(u >= 0.0, torch.ones_like(u), -torch.ones_like(u)) + 0.25 * v
