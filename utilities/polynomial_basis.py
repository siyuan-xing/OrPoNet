# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Polynomial basis evaluation and probability-measure normalization."""

import math

import numpy as np
import torch


def get_training_domain(polynomial_type, dim):
    """Return one standard training domain for the requested basis type."""
    if polynomial_type in ["legendre", "chebyshev"]:
        return [-1.0] * dim, [1.0] * dim
    if polynomial_type == "hermite":
        return [-3.0] * dim, [3.0] * dim
    return [-1.0] * dim, [1.0] * dim


def _get_basis_normalization_factor(poly_type, degree):
    """Normalize under each basis's probability measure.

    Legendre uses the uniform density on [-1, 1], Chebyshev uses
    1 / (pi * sqrt(1 - x**2)), and physicists' Hermite uses exp(-x**2)
    / sqrt(pi) on the real line.
    """
    if poly_type == "legendre":
        return np.sqrt(2 * degree + 1)

    if poly_type == "chebyshev":
        return 1.0 if degree == 0 else np.sqrt(2.0)

    if poly_type == "hermite":
        return 1.0 / np.sqrt((2**degree) * math.factorial(degree))

    raise ValueError(
        f"Basis normalization is not implemented for polynomial type: {poly_type}"
    )


class PolynomialFactory:
    """Evaluate polynomial bases through a shared three-term recurrence."""

    @staticmethod
    def build_polynomial_values(config, x, dtype=torch.float32, normalize_basis=False):
        """Evaluate degrees 0 through the requested degree by recurrence.

        Recurrences implementation. Custom functions are evaluated
        directly and do not use this recurrence.
        """
        poly_type = config["type"]
        degree = config["degree"]
        if poly_type == "custom":
            if normalize_basis:
                raise ValueError(
                    "Basis normalization is not implemented for custom polynomial functions."
                )
            return torch.cat([function(x) for function in config["functions"]], dim=1)
        if poly_type not in ("legendre", "hermite", "chebyshev"):
            raise ValueError(f"Unknown polynomial type: {poly_type}")

        x = x.to(dtype)
        columns = [torch.ones_like(x)]
        if degree > 0:
            columns.append(2 * x if poly_type == "hermite" else x)

        # At step n, previous and current contain degrees n-1 and n.
        # Use their unnormalized values to construct degree n+1.
        for n in range(1, degree):
            previous, current = columns[-2:]
            if poly_type == "legendre":
                # (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x).
                following = ((2 * n + 1) * x * current - n * previous) / (n + 1)
            elif poly_type == "hermite":
                # H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x).
                following = 2 * x * current - 2 * n * previous
            else:
                # T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x), first-kind Chebyshev.
                following = 2 * x * current - previous
            columns.append(following)

        values = torch.cat(columns, dim=1)
        # These formulas apply to unnormalized polynomials. Normalize only after
        # recurrence.
        if normalize_basis:
            factors = values.new_tensor(
                [
                    _get_basis_normalization_factor(poly_type, n)
                    for n in range(degree + 1)
                ]
            )
            values = values * factors
        return values
