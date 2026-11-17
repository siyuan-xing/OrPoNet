# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""OrPoNet for PyTorch.

"""

import numpy as np
import torch
import torch.nn as nn

from .polynomial_basis import PolynomialFactory


def _initialize_weight(shape, coeff_init_std, dtype):
    """Initialize coefficient weights uniformly with the requested standard deviation."""
    # A symmetric uniform distribution has standard deviation bound / sqrt(3).
    init_bound = np.sqrt(3.0) * coeff_init_std
    return torch.empty(shape, dtype=dtype).uniform_(-init_bound, init_bound)


def _build_marginalization_mask(feature_count, constant_channels, dtype, device):
    """Build the expectation mask for the current basis layout."""
    mask = torch.zeros((1, feature_count), dtype=dtype, device=device)
    mask[:, :constant_channels] = 1.0
    return mask


def _get_layer_width(polynomial_config):
    """Count all constant and nonconstant columns for one coordinate."""
    constant_channels = polynomial_config.get("constant_channels", 2)

    if polynomial_config["type"] == "custom":
        num_functions = len(polynomial_config["functions"])
    else:
        num_functions = polynomial_config["degree"] + 1
    return num_functions + constant_channels - 1


def _build_layer_basis(
    inputs, polynomial_config, dtype, normalize_basis=False, marginalize=False
):
    """Assemble the configured constant channels and polynomial basis."""
    polynomial_values = PolynomialFactory.build_polynomial_values(
        polynomial_config, inputs, dtype=dtype, normalize_basis=normalize_basis
    )
    # The polynomial basis already contains one constant (degree-zero) column.
    # Prepend only the additional constants; the default preserves the old layout.
    constant_channels = polynomial_config.get("constant_channels", 2)
    constant_term = torch.ones(
        (inputs.shape[0], constant_channels - 1), dtype=dtype, device=inputs.device
    )
    layer_basis = torch.cat([constant_term, polynomial_values], dim=1)
    if marginalize:
        layer_basis = layer_basis * _build_marginalization_mask(
            layer_basis.shape[1], constant_channels, dtype, inputs.device
        )
    return layer_basis


class OrPoNet(nn.Module):
    """OrPoNet with explicit copy channels."""

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
    ):
        super().__init__()

        self.dtype = dtype
        self._basis_configs = basis_configs
        self._hidden_layer_num = len(basis_configs)
        self.neuron_copies = neuron_copies
        self.normalize_basis = normalize_basis
        self._scaling_layer = ScalingLayer(
            data_domain_lb,
            data_domain_ub,
            training_domain_lb,
            training_domain_ub,
            dtype,
        )

        self._build_model(
            output_layer_neuron_num=output_layer_neuron_num,
            coeff_init_std=coeff_init_std,
        )

    @property
    def moment_propagation(self):
        """Access moment and Sobol analysis without storing extra model state."""
        # Import on access to avoid a circular dependency through residual layers.
        from .moment_propagation import MomentPropagation

        return MomentPropagation(self)

    def _build_model(self, output_layer_neuron_num, coeff_init_std):
        """Construct polynomial layers and output weights from the saved settings."""
        basis_configs = self._basis_configs
        # Build the first coordinate layer, followed by weighted polynomial layers.
        self._hidden_layers = nn.ModuleList()
        self._hidden_layers.append(
            FirstPolynomialLayer(
                basis_configs[0],
                dtype=self.dtype,
                neuron_copies=self.neuron_copies,
                normalize_basis=self.normalize_basis,
            )
        )

        prev_dim = self._hidden_layers[0].output_dim
        for layer_index in range(1, self._hidden_layer_num):
            layer = self._create_hidden_layer(
                basis_configs[layer_index], prev_dim, coeff_init_std
            )
            self._hidden_layers.append(layer)
            prev_dim = layer.output_dim

        # Combine the final polynomial features and copy channels into outputs.
        self._output_weight = nn.Parameter(
            _initialize_weight(
                shape=(output_layer_neuron_num, prev_dim, self.neuron_copies),
                coeff_init_std=coeff_init_std,
                dtype=self.dtype,
            )
        )

    def _create_hidden_layer(self, polynomial_config, input_dim, coeff_init_std):
        """Select the hidden layer; residual models override this factory."""
        return PolynomialLayer(
            polynomial_config,
            input_dim=input_dim,
            coeff_init_std=coeff_init_std,
            dtype=self.dtype,
            neuron_copies=self.neuron_copies,
            normalize_basis=self.normalize_basis,
        )

    def apply_basis_aware_init(self, reference_inputs, gain=0.1):
        """Initialize weights explicitly from a reference batch before training."""
        scaled_inputs = self._scaling_layer(reference_inputs).to(self.dtype)
        splitted_data = torch.split(scaled_inputs, 1, dim=-1)

        xn = self._hidden_layers[0](splitted_data[0])
        for layer_index in range(1, self._hidden_layer_num):
            layer = self._hidden_layers[layer_index]
            coord_comp = splitted_data[layer_index]

            base_polynomial_values = _build_layer_basis(
                coord_comp,
                layer.polynomial_config,
                layer.dtype,
                normalize_basis=layer.normalize_basis,
            )
            beta = torch.mean(base_polynomial_values**2, dim=0)
            fan_in = xn.shape[1]
            column_std = gain / torch.sqrt(fan_in * beta)
            layer.projection_w.data = torch.randn_like(
                layer.projection_w.data
            ) * column_std.view(1, -1, 1)

            xn = layer((xn, coord_comp))

        output_fan_in = xn.shape[1] * xn.shape[2]
        output_std = gain / np.sqrt(output_fan_in)
        self._output_weight.data = (
            torch.randn_like(self._output_weight.data) * output_std
        )

    def forward(self, inputs, marginalized_dims=None):
        if marginalized_dims is None:
            marginalized_dims = []
        marginalized_dims = set(marginalized_dims)

        scaled_inputs = self._scaling_layer(inputs).to(self.dtype)
        splitted_data = torch.split(scaled_inputs, 1, dim=-1)

        xn = self._hidden_layers[0](
            splitted_data[0], marginalize=0 in marginalized_dims
        )
        for layer_index in range(1, self._hidden_layer_num):
            xn = self._hidden_layers[layer_index](
                (xn, splitted_data[layer_index]),
                marginalize=layer_index in marginalized_dims,
            )

        outputs = torch.einsum("bic,oic->bo", xn, self._output_weight)
        return outputs


class ScalingLayer(nn.Module):
    """Scale inputs from the data domain to the training domain."""

    def __init__(self, data_lb, data_ub, training_lb, training_ub, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.register_buffer("data_lb", torch.tensor(data_lb, dtype=dtype))
        self.register_buffer("data_ub", torch.tensor(data_ub, dtype=dtype))
        self.register_buffer("training_lb", torch.tensor(training_lb, dtype=dtype))
        self.register_buffer("training_ub", torch.tensor(training_ub, dtype=dtype))
        self.register_buffer("data_range", self.data_ub - self.data_lb)
        self.register_buffer("training_range", self.training_ub - self.training_lb)

    def forward(self, inputs):
        inputs = inputs.to(self.dtype)
        normalized = (inputs - self.data_lb) / self.data_range
        scaled = normalized * self.training_range + self.training_lb
        return scaled


class FirstPolynomialLayer(nn.Module):
    """First polynomial layer with an explicit copy dimension."""

    def __init__(
        self,
        polynomial_config,
        dtype=torch.float32,
        neuron_copies=1,
        normalize_basis=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.polynomial_config = polynomial_config
        self.output_dim = _get_layer_width(polynomial_config)
        self.neuron_copies = neuron_copies
        self.normalize_basis = normalize_basis

    def forward(self, inputs, marginalize=False):
        inputs = inputs.to(self.dtype)
        base_output = _build_layer_basis(
            inputs,
            self.polynomial_config,
            self.dtype,
            normalize_basis=self.normalize_basis,
            marginalize=marginalize,
        )
        return base_output.unsqueeze(-1).repeat(1, 1, self.neuron_copies)


class PolynomialLayer(nn.Module):
    """Polynomial layer with per-copy weights stored in a 3D tensor."""

    def __init__(
        self,
        polynomial_config,
        input_dim,
        coeff_init_std=0.1,
        dtype=torch.float32,
        neuron_copies=1,
        normalize_basis=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.polynomial_config = polynomial_config
        self.output_dim = _get_layer_width(polynomial_config)
        self.neuron_copies = neuron_copies
        self.normalize_basis = normalize_basis

        self.projection_w = nn.Parameter(
            _initialize_weight(
                shape=(input_dim, self.output_dim, neuron_copies),
                coeff_init_std=coeff_init_std,
                dtype=dtype,
            )
        )

    def forward(self, inputs, marginalize=False):
        input_prev, coord_comp = inputs
        input_prev = input_prev.to(self.dtype)
        coord_comp = coord_comp.to(self.dtype)

        base_polynomial_values = _build_layer_basis(
            coord_comp,
            self.polynomial_config,
            self.dtype,
            normalize_basis=self.normalize_basis,
            marginalize=marginalize,
        )
        projected_input = torch.einsum("bic,ioc->boc", input_prev, self.projection_w)
        return projected_input * base_polynomial_values.unsqueeze(-1)
