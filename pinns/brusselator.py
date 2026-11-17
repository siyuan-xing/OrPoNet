# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Analytic forcing and an independent finite-difference Brusselator reference."""

import numpy as np
import torch
from scipy.sparse import bmat, diags, eye, kron
from scipy.sparse.linalg import spsolve


def brusselator_source(points):
    """Return the Figure 9 source terms as two columns on [-1, 1]^2."""
    x, y = points[:, :1], points[:, 1:2]
    source_u = (
        0.18 * torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * y)
        - 0.11 * torch.cos(2 * torch.pi * x) * torch.sin(torch.pi * y)
        + 0.07 * torch.cos(3 * torch.pi * x) * torch.cos(2 * torch.pi * y)
        + 0.05 * x * y
    )
    source_v = (
        -0.14 * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)
        + 0.09 * torch.sin(2 * torch.pi * x) * torch.sin(torch.pi * y)
        - 0.06 * torch.cos(2 * torch.pi * y)
        + 0.04 * (x.square() - 0.5 * y.square())
    )
    return torch.cat([source_u, source_v], dim=1)


def solve_brusselator_reference(grid_size, parameters):
    """Solve the steady PDE with central differences and damped sparse Newton.

    Work with deviations from the constant boundary values. This makes the
    boundary deviations zero, so the interior Laplacian needs no boundary RHS.
    The reference is independent of the PINN and is used only for evaluation.
    Return NumPy arrays axis, u, v, plus the final discrete residual norm.
    """
    if grid_size < 3:
        raise ValueError("The reference grid needs at least three points per axis.")
    a, b, du, dv = (parameters[key] for key in ["a", "b", "du", "dv"])
    axis = np.linspace(-1, 1, grid_size)
    spacing = axis[1] - axis[0]
    side = grid_size - 2
    count = side**2
    xx, yy = np.meshgrid(axis[1:-1], axis[1:-1], indexing="ij")
    points = torch.from_numpy(np.column_stack([xx.ravel(), yy.ravel()]))
    source = brusselator_source(points).numpy()

    # Five-point Laplacian in the same row-major order as the flattened fields.
    second_difference = (
        diags(
            [np.ones(side - 1), -2 * np.ones(side), np.ones(side - 1)],
            [-1, 0, 1],
            format="csr",
        )
        / spacing**2
    )
    identity = eye(side, format="csr")
    laplacian = kron(second_difference, identity) + kron(identity, second_difference)

    def residual(state):
        u, v = a + state[:count], b / a + state[count:]
        reaction = u**2 * v
        ru = (
            du * (laplacian @ state[:count]) + a - (b + 1) * u + reaction - source[:, 0]
        )
        rv = dv * (laplacian @ state[count:]) + b * u - reaction - source[:, 1]
        return np.concatenate([ru, rv])

    state = np.zeros(2 * count)
    for _ in range(20):
        current = residual(state)
        norm = np.linalg.norm(current)
        if norm < 1e-10:
            break
        u, v = a + state[:count], b / a + state[count:]
        jacobian = bmat(
            [
                [du * laplacian + diags(-(b + 1) + 2 * u * v), diags(u**2)],
                [diags(b - 2 * u * v), dv * laplacian - diags(u**2)],
            ],
            format="csr",
        )
        update = spsolve(jacobian, -current)
        # Backtrack until Newton decreases the residual, as in the original solver.
        step = 1.0
        while step >= 1e-4:
            candidate = state + step * update
            if np.linalg.norm(residual(candidate)) < norm:
                state = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("The reference Newton line search did not converge.")

    final_norm = np.linalg.norm(residual(state))
    if final_norm >= 1e-10:
        raise RuntimeError(
            "The reference Newton solve did not reach its residual tolerance."
        )
    u = np.full((grid_size, grid_size), a, dtype=np.float64)
    v = np.full((grid_size, grid_size), b / a, dtype=np.float64)
    u[1:-1, 1:-1] += state[:count].reshape(side, side)
    v[1:-1, 1:-1] += state[count:].reshape(side, side)
    return {"axis": axis, "u": u, "v": v, "residual_norm": final_norm}


class BrusselatorPINN(torch.nn.Module):
    """Two OrPoNet cores with the Figure 9/11 boundary values enforced exactly."""

    def __init__(self, degree, parameters, dtype=torch.float64):
        super().__init__()
        from utilities.orponet import OrPoNet

        settings = dict(
            basis_configs=[
                {"type": "chebyshev", "degree": degree, "constant_channels": 2}
                for _ in range(2)
            ],
            data_domain_lb=[-1, -1],
            data_domain_ub=[1, 1],
            training_domain_lb=[-1, -1],
            training_domain_ub=[1, 1],
            coeff_init_std=0.05,
            neuron_copies=1,
            normalize_basis=False,
            dtype=dtype,
        )
        self.u_model = OrPoNet(**settings)
        self.v_model = OrPoNet(**settings)
        self.register_buffer(
            "boundary_values",
            torch.tensor(
                [parameters["a"], parameters["b"] / parameters["a"]], dtype=dtype
            ),
        )

    def forward(self, points):
        x, y = points[:, :1], points[:, 1:2]
        factor = (1 - x.square()) * (1 - y.square())
        corrections = torch.cat([self.u_model(points), self.v_model(points)], dim=1)
        return self.boundary_values + factor * corrections


def brusselator_residual(model, points, parameters):
    """Return both steady PDE residuals, retaining second-derivative graphs."""
    points = points.detach().requires_grad_(True)

    def gradient(values):
        return torch.autograd.grad(
            values, points, torch.ones_like(values), create_graph=True
        )[0]

    def laplacian(values):
        first = gradient(values)
        return gradient(first[:, :1])[:, :1] + gradient(first[:, 1:2])[:, 1:2]

    prediction = model(points)
    u, v = prediction[:, :1], prediction[:, 1:2]
    source = brusselator_source(points)
    a, b, du, dv = (parameters[key] for key in ["a", "b", "du", "dv"])
    reaction = u.square() * v
    ru = du * laplacian(u) + a - (b + 1) * u + reaction - source[:, :1]
    rv = dv * laplacian(v) + b * u - reaction - source[:, 1:2]
    return torch.cat([ru, rv], dim=1)
