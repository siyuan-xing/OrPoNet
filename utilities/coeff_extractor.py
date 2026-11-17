# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Extract and evaluate full tensor-product expansions for low-dimensional models.

This algorithm propagates coefficient tensors recursively through the network.
Use this module for low-dimensional coefficient recovery and interpretation.
"""

import itertools
import math

import torch

from .polynomial_basis import PolynomialFactory
from .residual_orponet import ResidualPolynomialLayer


class CoefficientExtractor:
    """Read polynomial coefficients directly from an OrPoNet or ResidualOrPoNet.

    Indices are actual polynomial degrees, starting at zero. All constant
    channels are merged into degree zero. The expansion uses the model's
    scaled coordinates and the same basis normalization as the model.
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def extract_coefficients(self, max_terms=100_000):
        """Return coefficients (terms, outputs) and degree indices (terms, dimensions).

        Full expansion grows as the product of (degree+1). The term limit is
        checked before allocating tensors; high-dimensional moment and Sobol
        calculations should avoid this full expansion.
        """
        configs = self.model._basis_configs
        for config in configs:
            if config["type"] not in ("legendre", "hermite", "chebyshev"):
                raise ValueError(
                    "Coefficient extraction requires a built-in polynomial basis."
                )
        term_count = math.prod(config["degree"] + 1 for config in configs)
        if term_count > max_terms:
            raise ValueError(
                f"Full expansion needs {term_count} terms, exceeding max_terms={max_terms}."
            )

        # Map raw features onto distinct polynomial degrees, combining constants.
        mapping = self._basis_mapping(configs[0])
        state = mapping.T.unsqueeze(-1).expand(-1, -1, self.model.neuron_copies)
        for config, layer in zip(configs[1:], self.model._hidden_layers[1:]):
            mapping = self._basis_mapping(config)
            projected = torch.einsum("tic,ioc->toc", state, layer.projection_w)
            # Append the new coordinate's degree to each existing multi-index.
            expanded = projected[:, None] * mapping.T[None, :, :, None]
            if isinstance(layer, ResidualPolynomialLayer):
                # Skip paths do not depend on the new coordinate: degree zero.
                skip = state
                if layer.skip_projection is not None:
                    skip = torch.einsum("tic,ioc->toc", state, layer.skip_projection)
                gate = layer.get_residual_gate()
                if gate is not None:
                    skip = skip * gate.unsqueeze(0)
                expanded[:, 0] += skip
                if layer.basis_projection is not None:
                    # The additive branch has degree zero in every earlier coordinate.
                    additive = torch.einsum(
                        "jq,joc->qoc", mapping, layer.basis_projection
                    )
                    expanded[0] += additive
            state = expanded.reshape(-1, layer.output_dim, self.model.neuron_copies)

        coefficients = torch.einsum("tic,oic->to", state, self.model._output_weight)
        indices = list(
            itertools.product(*(range(config["degree"] + 1) for config in configs))
        )
        multi_indices = torch.tensor(
            indices, dtype=torch.long, device=coefficients.device
        )
        return coefficients, multi_indices

    def _basis_mapping(self, config):
        """Map each raw basis column to its canonical degree without normalization changes."""
        constants = config.get("constant_channels", 2)
        degree = config["degree"]
        mapping = self.model._output_weight.new_zeros((degree + constants, degree + 1))
        mapping[:constants, 0] = 1
        for n in range(1, degree + 1):
            mapping[constants + n - 1, n] = 1
        return mapping

    def evaluate_pce(self, inputs, coefficients, multi_indices):
        """Reconstruct predictions at physical inputs from an extracted expansion.

        This applies the model's coordinate scaling. Coefficients must use
        the basis families and normalization of this extractor's model.
        """
        coordinates = self.model._scaling_layer(inputs).to(self.model.dtype)
        terms = coordinates.new_ones((coordinates.shape[0], multi_indices.shape[0]))
        for dimension, config in enumerate(self.model._basis_configs):
            values = PolynomialFactory.build_polynomial_values(
                config,
                coordinates[:, dimension : dimension + 1],
                dtype=self.model.dtype,
                normalize_basis=self.model.normalize_basis,
            )
            terms = terms * values[:, multi_indices[:, dimension]]
        return terms @ coefficients
