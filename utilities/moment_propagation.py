# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Weight-space moment propagation, extracted from the existing research tool.

Assumptions: scalar output, independent coordinates, normalize_basis=True, and
each scaled coordinate follows its basis's orthogonality probability measure:
uniform [-1,1] for Legendre, arcsine on [-1,1] for Chebyshev, and exp(-x*x)/sqrt(pi)
on the real line for physicists' Hermite. Uniformly sampled Hermite/Chebyshev
inputs generally do not satisfy these measure assumptions.
"""

import torch

from .residual_orponet import ResidualPolynomialLayer


class MomentPropagation:
    """Analyze scalar OrPoNet models using the existing moment-propagation formulas."""

    def __init__(self, model):
        self.model = model
        self.dim = len(model._basis_configs)
        self.basis_configs = model._basis_configs
        self.neuron_copies = model.neuron_copies
        if model._output_weight.shape[0] != 1:
            raise ValueError(
                "MomentPropagation currently requires a scalar output model."
            )

    @property
    def device(self):
        # Read dynamically so an existing analyzer follows model.to(device).
        return self.model._output_weight.device

    @torch.no_grad()
    def moments(self, active_dims=None):
        """Return mean and variance after averaging out all inactive coordinates.

        None retains all coordinates; an empty list retains none and returns
        the constant global mean. All results are float64 tensors on the model's
        device. Weights and gates are read at call time; no gradients are built.
        """
        active_dims = set(range(self.dim)) if active_dims is None else set(active_dims)
        mean, second = self._output_moments(active_dims)
        return {"mean": mean, "second_moment": second, "variance": second - mean**2}

    def _build_raw_basis_moments(
        self, feature_count, marginalized, constant_channels=2
    ):
        """
        Build the first and second raw moments of one layer basis layout.

        The first constant_channels entries are identical constants, including
        the degree-zero polynomial. Their complete second-moment block is 1.

        With normalized bases, all higher-order channels are orthonormal and
        zero-mean. The second-moment matrix therefore has one duplicated
        constant block plus identity on the non-constant basis channels.
        """
        # Promote analysis to float64 without modifying the model's dtype/device.
        mean = torch.zeros(feature_count, dtype=torch.float64, device=self.device)
        mean[:constant_channels] = 1.0

        second = torch.zeros(
            (feature_count, feature_count), dtype=torch.float64, device=self.device
        )
        second[:constant_channels, :constant_channels] = 1.0
        if not marginalized:
            for basis_index in range(constant_channels, feature_count):
                second[basis_index, basis_index] = 1.0

        return mean, second

    def _first_layer_moments(self, active_dims):
        """Compute the mean and second moments of the first layer's output."""
        first_feature_count = self.model._hidden_layers[0].output_dim
        mean_basis, second_basis = self._build_raw_basis_moments(
            first_feature_count,
            marginalized=0 not in active_dims,
            constant_channels=self.basis_configs[0].get("constant_channels", 2),
        )

        mean = mean_basis[:, None].expand(-1, self.neuron_copies)
        # Axes: (left feature, left copy, right feature, right copy).
        # Copies share the same input, so every copy pair has the same moments.
        second = (
            second_basis[:, None, :, None]
            .expand(
                first_feature_count,
                self.neuron_copies,
                first_feature_count,
                self.neuron_copies,
            )
            .clone()
        )
        return mean, second

    def _first_order_moment(self, mean_prev, weight_tensor):
        """Compute W.T @ mean for each copy; input axes are (feature, copy)."""
        output_dim = weight_tensor.shape[1]
        mean_next = torch.zeros(
            (output_dim, self.neuron_copies), dtype=torch.float64, device=self.device
        )
        for copy_index in range(self.neuron_copies):
            mean_next[:, copy_index] = (
                weight_tensor[:, :, copy_index].T @ mean_prev[:, copy_index]
            )
        return mean_next

    def _second_order_moment(self, second_prev, left_weight, right_weight):
        """Compute W_left.T @ second @ W_right for every copy pair.

        Input axes: (left feature, left copy, right feature, right copy).
        For different left/right weights this also computes cross moments.
        """
        output_dim = left_weight.shape[1]
        second_next = torch.zeros(
            (
                output_dim,
                self.neuron_copies,
                output_dim,
                self.neuron_copies,
            ),
            dtype=torch.float64,
            device=self.device,
        )
        for copy_left in range(self.neuron_copies):
            left_matrix = left_weight[:, :, copy_left]
            for copy_right in range(self.neuron_copies):
                right_matrix = right_weight[:, :, copy_right]
                second_next[:, copy_left, :, copy_right] = (
                    left_matrix.T
                    @ second_prev[:, copy_left, :, copy_right]
                    @ right_matrix
                )
        return second_next

    def _propagate_standard_layer(
        self, mean_prev, second_prev, layer_index, active_dims
    ):
        """Propagate hidden moments through one standard multiplicative layer."""
        feature_count = self.model._hidden_layers[layer_index].output_dim
        basis_mean, basis_second = self._build_raw_basis_moments(
            feature_count,
            marginalized=layer_index not in active_dims,
            constant_channels=self.basis_configs[layer_index].get(
                "constant_channels", 2
            ),
        )
        projection = (
            self.model._hidden_layers[layer_index]
            .projection_w.detach()
            .to(dtype=torch.float64)
        )

        mean_projected = self._first_order_moment(mean_prev, projection)
        second_projected = self._second_order_moment(
            second_prev, projection, projection
        )

        mean_next = mean_projected * basis_mean[:, None]
        # Broadcast basis moments over both copy axes.
        second_next = second_projected * basis_second[:, None, :, None]

        return mean_next, second_next

    def _propagate_residual_layer(
        self, mean_prev, second_prev, layer_index, active_dims
    ):
        """Propagate hidden moments through one residual layer exactly in weight space."""
        hidden_layer = self.model._hidden_layers[layer_index]
        feature_count = self.model._hidden_layers[layer_index].output_dim
        basis_mean, basis_second = self._build_raw_basis_moments(
            feature_count,
            marginalized=layer_index not in active_dims,
            constant_channels=self.basis_configs[layer_index].get(
                "constant_channels", 2
            ),
        )

        projection = hidden_layer.polynomial_layer.projection_w.detach().to(
            dtype=torch.float64
        )

        if hidden_layer.skip_projection is None:
            skip_projection = torch.eye(
                mean_prev.shape[0],
                feature_count,
                dtype=torch.float64,
                device=self.device,
            )[:, :, None].expand(-1, -1, self.neuron_copies)
        else:
            skip_projection = hidden_layer.skip_projection.detach().to(
                dtype=torch.float64
            )

        # Incorporate the gate into every skip moment, including cross moments.
        gate = hidden_layer.get_residual_gate()
        if gate is not None:
            skip_projection = (
                skip_projection * gate.detach().to(dtype=torch.float64)[None, :, :]
            )

        mean_skip = self._first_order_moment(mean_prev, skip_projection)
        second_skip = self._second_order_moment(
            second_prev, skip_projection, skip_projection
        )
        mean_mult_base = self._first_order_moment(mean_prev, projection)
        second_mult_base = self._second_order_moment(
            second_prev, projection, projection
        )

        mean_mult = mean_mult_base * basis_mean[:, None]
        second_mult = second_mult_base * basis_second[:, None, :, None]

        # E[s t^T]: only the multiplicative (right) branch contains the basis.
        cross_skip_mult = self._second_order_moment(
            second_prev, skip_projection, projection
        )
        cross_skip_mult *= basis_mean[None, None, :, None]
        mean_next = mean_skip + mean_mult
        second_next = second_skip + second_mult
        # Swapping the left/right feature-copy pairs gives E[t s^T].
        second_next += cross_skip_mult + cross_skip_mult.permute(2, 3, 0, 1)

        if hidden_layer.basis_projection is not None:
            basis_projection = hidden_layer.basis_projection.detach().to(
                dtype=torch.float64
            )
            mean_additive = self._first_order_moment(
                basis_mean[:, None].expand(-1, self.neuron_copies),
                basis_projection,
            )
            # Every copy uses the same current-coordinate moment matrix.
            # expand adds copy axes as a view, without copying the matrix.
            shared_second = basis_second[:, None, :, None].expand(
                feature_count,
                self.neuron_copies,
                feature_count,
                self.neuron_copies,
            )
            second_additive = self._second_order_moment(
                shared_second,
                basis_projection,
                basis_projection,
            )

            # Skip depends on previous coordinates; the basis branch on this one.
            cross_skip_additive = (
                mean_skip[:, :, None, None] * mean_additive[None, None, :, :]
            )

            # E[t a^T] = E[u] * E[b (B^T b)^T]. Retain all copy correlations.
            cross_mult_additive = torch.empty_like(second_additive)
            for copy_right in range(self.neuron_copies):
                basis_cross = basis_second @ basis_projection[:, :, copy_right]
                cross_mult_additive[:, :, :, copy_right] = (
                    mean_mult_base[:, :, None] * basis_cross[:, None, :]
                )

            mean_next += mean_additive
            second_next += second_additive
            second_next += cross_skip_additive + cross_skip_additive.permute(2, 3, 0, 1)
            second_next += cross_mult_additive + cross_mult_additive.permute(2, 3, 0, 1)

        return mean_next, second_next

    def _output_moments(self, active_dims):
        """Compute output mean and second moment directly from network weights."""
        if not getattr(self.model, "normalize_basis", False):
            raise ValueError(
                "Weight-space Sobol extraction requires normalize_basis=True."
            )

        mean_hidden, second_hidden = self._first_layer_moments(active_dims)
        for layer_index in range(1, self.dim):
            hidden_layer = self.model._hidden_layers[layer_index]
            if isinstance(hidden_layer, ResidualPolynomialLayer):
                mean_hidden, second_hidden = self._propagate_residual_layer(
                    mean_hidden,
                    second_hidden,
                    layer_index,
                    active_dims,
                )
            else:
                mean_hidden, second_hidden = self._propagate_standard_layer(
                    mean_hidden,
                    second_hidden,
                    layer_index,
                    active_dims,
                )

        output_weight = (
            self.model._output_weight.detach().to(dtype=torch.float64).reshape(-1)
        )
        mean_scalar = mean_hidden.reshape(-1) @ output_weight
        second_matrix = second_hidden.reshape(
            output_weight.numel(), output_weight.numel()
        )
        second_scalar = output_weight @ second_matrix @ output_weight
        return mean_scalar, second_scalar

    @torch.no_grad()
    def sobol_from_weights(self):
        """
        Compute Sobol indices directly from network weights using moment propagation.

        This path keeps the computation in the raw hidden-feature space. For each
        Sobol quantity we select one set of active dimensions and marginalize the
        rest, then propagate output mean and variance through the network exactly.
        """
        full = self.moments()
        total_variance = full["variance"]
        if total_variance <= 0.0:
            raise ValueError(
                "The extracted network has zero variance, so Sobol indices are undefined."
            )

        first_order = torch.zeros(self.dim, dtype=torch.float64, device=self.device)
        total_order = torch.zeros(self.dim, dtype=torch.float64, device=self.device)
        all_dims = set(range(self.dim))
        for dim_index in range(self.dim):
            # S_i = Var(E[Y | X_i]) / Var(Y).
            first = self.moments(active_dims=[dim_index])
            first_order[dim_index] = first["variance"].clamp_min(0.0) / total_variance

            # ST_i = 1 - Var(E[Y | X_{-i}]) / Var(Y).
            complement = self.moments(active_dims=all_dims - {dim_index})
            total_order[dim_index] = (
                1.0 - complement["variance"].clamp_min(0.0) / total_variance
            )

        return {
            "mean": full["mean"],
            "variance": total_variance,
            "first_order": first_order,
            "total_order": total_order,
        }
