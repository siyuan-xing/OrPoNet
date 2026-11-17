# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Residual OrPoNet"""

import torch
import torch.nn as nn

from .orponet import OrPoNet, PolynomialLayer, _build_layer_basis, _initialize_weight


class ResidualOrPoNet(OrPoNet):
    """OrPoNet with skip connections, an additive basis branch, and optional gates.

    Forward evaluation, scaling, per-layer constant channels, and explicit
    basis-aware initialization reuse OrPoNet. Basis-aware initialization resets
    multiplicative and output coefficients; skip, additive, and gate parameters
    retain their current values, matching the paper implementation.
    """

    def __init__(
        self,
        basis_configs,
        data_domain_lb=None,
        data_domain_ub=None,
        training_domain_lb=None,
        training_domain_ub=None,
        output_layer_neuron_num=1,
        coeff_init_std=0.1,
        dtype=torch.float32,
        neuron_copies=1,
        normalize_basis=False,
        use_additive_basis_branch=True,
        gate_config=None,
    ):
        # Save the residual settings before the parent constructs the layers.
        self.use_additive_basis_branch = use_additive_basis_branch
        # None disables gates; even an empty dictionary enables default gates.
        self.gate_config = None if gate_config is None else dict(gate_config)
        super().__init__(
            basis_configs=basis_configs,
            data_domain_lb=data_domain_lb,
            data_domain_ub=data_domain_ub,
            training_domain_lb=training_domain_lb,
            training_domain_ub=training_domain_ub,
            output_layer_neuron_num=output_layer_neuron_num,
            coeff_init_std=coeff_init_std,
            dtype=dtype,
            neuron_copies=neuron_copies,
            normalize_basis=normalize_basis,
        )

    def _create_hidden_layer(self, polynomial_config, input_dim, coeff_init_std):
        """Use a residual layer within the shared model-construction workflow."""
        return ResidualPolynomialLayer(
            polynomial_config,
            input_dim=input_dim,
            coeff_init_std=coeff_init_std,
            dtype=self.dtype,
            neuron_copies=self.neuron_copies,
            normalize_basis=self.normalize_basis,
            use_additive_basis_branch=self.use_additive_basis_branch,
            gate_config=self.gate_config,
        )

    def residual_gate_l1(self):
        """Return the mean gate-open probability for sparsity regularization."""
        probabilities = [
            layer.residual_gate_probability() for layer in self._hidden_layers[1:]
        ]
        penalties = [
            probability.mean()
            for probability in probabilities
            if probability is not None
        ]
        if not penalties:
            return self._output_weight.new_zeros(())
        return torch.stack(penalties).mean()

    def set_residual_gate_force_open(self, force_open):
        """Temporarily force all residual gates open without changing their logits."""
        for layer in self._hidden_layers[1:]:
            layer.force_residual_gate_open = bool(force_open)


class ResidualPolynomialLayer(nn.Module):
    """Residual polynomial layer with explicit copy channels."""

    def __init__(
        self,
        polynomial_config,
        input_dim,
        coeff_init_std=0.1,
        dtype=torch.float32,
        neuron_copies=1,
        normalize_basis=False,
        use_additive_basis_branch=True,
        gate_config=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.polynomial_config = polynomial_config
        self.normalize_basis = normalize_basis
        self.use_additive_basis_branch = use_additive_basis_branch
        self.force_residual_gate_open = False
        self.polynomial_layer = PolynomialLayer(
            polynomial_config=polynomial_config,
            input_dim=input_dim,
            coeff_init_std=coeff_init_std,
            dtype=dtype,
            neuron_copies=neuron_copies,
            normalize_basis=normalize_basis,
        )

        output_dim = self.polynomial_layer.output_dim
        self.output_dim = output_dim
        self.neuron_copies = neuron_copies
        if gate_config is not None:
            # Read defaults only when gates are enabled.
            self.gate_temperature = float(gate_config.get("temperature", 1.0))
            self.gate_threshold = float(gate_config.get("threshold", 0.5))
            self.gate_mode = gate_config.get("mode", "hard")
            self.residual_gate_logit = nn.Parameter(
                torch.full(
                    (output_dim, neuron_copies),
                    float(gate_config.get("init_logit", 0.0)),
                    dtype=dtype,
                )
            )
        else:
            self.residual_gate_logit = None

        # Match unequal feature widths with a learned per-copy skip projection.
        if input_dim == output_dim:
            self.skip_projection = None
        else:
            self.skip_projection = nn.Parameter(
                _initialize_weight(
                    shape=(input_dim, output_dim, neuron_copies),
                    coeff_init_std=coeff_init_std,
                    dtype=dtype,
                )
            )

        if self.use_additive_basis_branch:
            self.basis_projection = nn.Parameter(
                _initialize_weight(
                    shape=(output_dim, output_dim, neuron_copies),
                    coeff_init_std=coeff_init_std,
                    dtype=dtype,
                )
            )
        else:
            self.basis_projection = None

    @property
    def projection_w(self):
        """Expose the multiplicative coefficients to the shared initializer."""
        return self.polynomial_layer.projection_w

    def residual_gate_probability(self):
        if self.residual_gate_logit is None:
            return None
        return torch.sigmoid(self.residual_gate_logit / self.gate_temperature)

    def get_residual_gate(self, hard=None):
        probability = self.residual_gate_probability()
        if probability is None:
            return None
        if self.force_residual_gate_open:
            return torch.ones_like(probability)
        if hard is None:
            hard = self.gate_mode == "hard"
        if not hard:
            return probability
        hard_gate = (probability >= self.gate_threshold).to(dtype=probability.dtype)
        if self.training:
            # Straight-through estimator: hard forward values, sigmoid gradients.
            return hard_gate + probability - probability.detach()
        return hard_gate

    def forward(self, inputs, marginalize=False):
        input_prev, coord_comp = inputs
        input_prev = input_prev.to(self.dtype)
        coord_comp = coord_comp.to(self.dtype)

        # Combine the tensor-product update with a skip and optional local basis branch.
        multiplicative_update = self.polynomial_layer(
            (input_prev, coord_comp), marginalize=marginalize
        )
        if self.use_additive_basis_branch:
            base_polynomial_values = _build_layer_basis(
                coord_comp,
                self.polynomial_config,
                self.dtype,
                normalize_basis=self.polynomial_layer.normalize_basis,
                marginalize=marginalize,
            )
            additive_basis_update = torch.einsum(
                "bi,ioc->boc", base_polynomial_values, self.basis_projection
            )
        else:
            additive_basis_update = 0.0

        if self.skip_projection is None:
            skip = input_prev
        else:
            skip = torch.einsum("bic,ioc->boc", input_prev, self.skip_projection)
        if self.residual_gate_logit is not None:
            skip = skip * self.get_residual_gate().unsqueeze(0)

        return skip + multiplicative_update + additive_basis_update
