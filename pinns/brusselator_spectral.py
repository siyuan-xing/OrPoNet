# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""First-party Chebyshev collocation baseline for the Figure 11 order study."""

import numpy as np
import torch
from scipy.optimize import least_squares

from .brusselator import brusselator_source
from utilities.polynomial_basis import PolynomialFactory


def _chebyshev_derivatives(x, degree):
    """Reuse basis values; differentiate the Chebyshev recurrence for derivatives."""
    values = PolynomialFactory.build_polynomial_values(
        {"type": "chebyshev", "degree": degree}, x, dtype=x.dtype
    )
    first = [torch.zeros_like(x)]
    second = [torch.zeros_like(x)]
    if degree > 0:
        first.append(torch.ones_like(x))
        second.append(torch.zeros_like(x))
    for n in range(1, degree):
        first.append(2 * values[:, n : n + 1] + 2 * x * first[n] - first[n - 1])
        second.append(4 * first[n] + 2 * x * second[n] - second[n - 1])
    return values, torch.cat(first, dim=1), torch.cat(second, dim=1)


def _lifted_basis(points, degree):
    """Return matrices for L*T_i(x)*T_j(y) and their exact Laplacians."""
    x, y = points[:, :1], points[:, 1:2]
    bx, dx, dxx = _chebyshev_derivatives(x, degree)
    by, dy, dyy = _chebyshev_derivatives(y, degree)

    def tensor_product(left, right):
        return (left[:, :, None] * right[:, None, :]).flatten(start_dim=1)

    basis = tensor_product(bx, by)
    lifting = (1 - x.square()) * (1 - y.square())
    lifting_x = -2 * x * (1 - y.square())
    lifting_y = -2 * y * (1 - x.square())
    lifting_laplacian = -2 * (1 - y.square()) - 2 * (1 - x.square())
    laplacian = (
        lifting_laplacian * basis
        + 2 * lifting_x * tensor_product(dx, by)
        + 2 * lifting_y * tensor_product(bx, dy)
        + lifting * (tensor_product(dxx, by) + tensor_product(bx, dyy))
    )
    return lifting * basis, laplacian


def solve_brusselator_spectral(degree, parameters, max_nfev=300):
    """Fit tensor-product Chebyshev coefficients by nonlinear least squares.

    Use degree+1 interior Lobatto nodes per coordinate, zero initial deviations
    from the constant boundary equilibrium, and the original 1e-11 tolerances.
    The analytic coefficient Jacobian avoids finite differences of every weight.
    Return coefficients and explicit solver status; hitting the budget is not
    treated as convergence. SciPy optimization runs on CPU in float64.
    """
    nodes = torch.cos(
        torch.pi * torch.arange(1, degree + 2, dtype=torch.float64) / (degree + 2)
    )
    points = torch.cartesian_prod(nodes, nodes)
    basis, laplacian = (matrix.numpy() for matrix in _lifted_basis(points, degree))
    source = brusselator_source(points).numpy()
    a, b, du, dv = (parameters[key] for key in ["a", "b", "du", "dv"])
    count = basis.shape[1]

    def fields(vector):
        u = a + basis @ vector[:count]
        v = b / a + basis @ vector[count:]
        return u, v

    def residual(vector):
        u, v = fields(vector)
        reaction = u**2 * v
        ru = (
            du * (laplacian @ vector[:count])
            + a
            - (b + 1) * u
            + reaction
            - source[:, 0]
        )
        rv = dv * (laplacian @ vector[count:]) + b * u - reaction - source[:, 1]
        return np.concatenate([ru, rv])

    def jacobian(vector):
        u, v = fields(vector)
        return np.block(
            [
                [
                    du * laplacian + (-(b + 1) + 2 * u * v)[:, None] * basis,
                    u[:, None] ** 2 * basis,
                ],
                [
                    (b - 2 * u * v)[:, None] * basis,
                    dv * laplacian - u[:, None] ** 2 * basis,
                ],
            ]
        )

    result = least_squares(
        residual,
        np.zeros(2 * count),
        jac=jacobian,
        max_nfev=max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    return {
        "degree": degree,
        "coefficients": torch.from_numpy(result.x.reshape(2, -1).T.copy()),
        "success": bool(result.success),
        "nfev": result.nfev,
        "message": result.message,
    }


def brusselator_spectral_residual(solution, points, parameters):
    """Evaluate both residuals on any grid, using the same convention as the PINN."""
    basis, laplacian = _lifted_basis(points, solution["degree"])
    coefficients = solution["coefficients"].to(dtype=points.dtype, device=points.device)
    a, b, du, dv = (parameters[key] for key in ["a", "b", "du", "dv"])
    fields = points.new_tensor([a, b / a]) + basis @ coefficients
    u, v = fields[:, :1], fields[:, 1:2]
    lap_u, lap_v = (laplacian @ coefficients).split(1, dim=1)
    source = brusselator_source(points)
    reaction = u.square() * v
    return torch.cat(
        [
            du * lap_u + a - (b + 1) * u + reaction - source[:, :1],
            dv * lap_v + b * u - reaction - source[:, 1:2],
        ],
        dim=1,
    )
