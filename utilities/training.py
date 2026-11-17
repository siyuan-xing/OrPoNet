# -----------------------------------------------------------------------------
# OrPoNet | Orthogonal Polynomial Network
# Author: Siyuan Xing
# ORCID: https://orcid.org/0000-0003-4502-3006
# Copyright (c) 2026 Siyuan Xing
# -----------------------------------------------------------------------------

"""Shared optimization and error histories; experiment recipes stay in notebooks."""

from copy import deepcopy

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    """Train a scalar-output model and record errors on the original target scale.

    train_data and validation_data are (inputs, original_targets) pairs.
    target_stats optionally supplies (training_mean, training_std); calculate
    those statistics in the notebook to retain its normalization convention.
    selection=None disables checkpoint selection. Otherwise select by a history
    field, usually validation_relative_l2 or validation_loss (standardized MSE).
    validation_batches optionally supplies a DataLoader of standardized targets;
    its mean of batch losses preserves the Figure 7 checkpoint-selection rule.
    """

    def __init__(
        self,
        model,
        train_data,
        validation_data,
        target_stats=None,
        selection="validation_relative_l2",
        validation_batches=None,
    ):
        self.model = model
        self.train_x, self.train_y = train_data
        self.validation_x, self.validation_y = validation_data
        self.target_mean, self.target_std = (
            (0.0, 1.0) if target_stats is None else target_stats
        )
        dtype = next(model.parameters()).dtype
        self.train_targets = ((self.train_y - self.target_mean) / self.target_std).to(
            dtype
        )
        self.validation_targets = (
            (self.validation_y - self.target_mean) / self.target_std
        ).to(dtype)
        self.selection = selection
        self.validation_batches = validation_batches
        self._history = []
        self._updates = 0
        self._best_score = float("inf")
        self._best_state = None

    @property
    def history(self):
        """Return accumulated per-update metrics as a DataFrame for plotting."""
        return pd.DataFrame(self._history)

    def _original_scale(self, prediction, targets):
        # Promote before inverse scaling, preserving Bratley's float64 targets.
        return prediction.to(targets.dtype) * self.target_std + self.target_mean

    @torch.no_grad()
    def _record(self, stage, select=True):
        self.model.eval()
        if self.validation_batches is None:
            train_prediction = self.model(self.train_x)
            validation_prediction = self.model(self.validation_x)
            validation_loss = (
                (validation_prediction - self.validation_targets).square().mean().item()
            )
        else:
            # Average validation batch losses for checkpoint selection.
            predictions = []
            validation_loss = 0.0
            for inputs, targets in self.validation_batches:
                prediction = self.model(inputs)
                validation_loss += (prediction - targets).square().mean().item()
                predictions.append(prediction)
            validation_loss /= len(self.validation_batches)
            validation_prediction = torch.cat(predictions)
            train_prediction = self.model(self.train_x)

        train_residual = (
            self._original_scale(train_prediction, self.train_y) - self.train_y
        )
        validation_residual = (
            self._original_scale(validation_prediction, self.validation_y)
            - self.validation_y
        )
        row = {
            "update": self._updates,
            "stage": stage,
            "train_mse": train_residual.square().mean().item(),
            "validation_mse": validation_residual.square().mean().item(),
            "train_relative_l2": (
                torch.linalg.vector_norm(train_residual)
                / torch.linalg.vector_norm(self.train_y)
            ).item(),
            "validation_relative_l2": (
                torch.linalg.vector_norm(validation_residual)
                / torch.linalg.vector_norm(self.validation_y)
            ).item(),
            "validation_loss": validation_loss,
        }
        self._history.append(row)
        improved = (
            select
            and self.selection is not None
            and row[self.selection] < self._best_score
        )
        if improved:
            self._best_score = row[self.selection]
            self._best_state = deepcopy(self.model.state_dict())
        return improved

    def record_initial(self):
        """Record update zero for a learning curve, without selecting a checkpoint."""
        if self._history:
            raise ValueError("Record the initial state before training.")
        self._record("initial", select=False)

    def fit(
        self,
        optimizer,
        epochs,
        batch_size=128,
        batches=None,
        initialize=None,
        early_stopping=None,
        log_every=None,
        l1_weight=0.0,
        record_every=1,
    ):
        """Perform one optimizer update per minibatch and record each epoch.

        By default use a shuffled DataLoader. batches optionally supplies a
        DataLoader of (inputs, standardized_targets). initialize(first_inputs) runs
        once before this fit's first update, after the first batch is drawn.
        early_stopping={"patience": ..., "grace": ...} counts bad epochs from
        the zero-based grace boundary and stops when bad_epochs > patience.
        l1_weight adds the sum of absolute parameter values to the objective;
        histories still report data errors. record_every controls evaluation and
        checkpoint frequency, always including the first and final epoch.
        Early stopping requires per-epoch recording. Checkpoints span calls;
        restore_best() is always explicit.
        """
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if record_every < 1:
            raise ValueError("record_every must be positive.")
        if early_stopping is not None and record_every != 1:
            raise ValueError("Early stopping requires record_every=1.")
        if early_stopping is not None and self.selection is None:
            raise ValueError("Early stopping requires a checkpoint-selection metric.")
        if batches is None:
            batches = DataLoader(
                TensorDataset(self.train_x, self.train_targets),
                batch_size=batch_size,
                shuffle=True,
            )
        if len(batches) == 0:
            raise ValueError("Training requires at least one batch.")
        bad_epochs = 0
        stage = optimizer.__class__.__name__.lower()
        for epoch in range(epochs):
            self.model.train()
            for batch_index, (inputs, targets) in enumerate(batches):
                if epoch == 0 and batch_index == 0 and initialize is not None:
                    initialize(inputs)
                optimizer.zero_grad()
                loss = (self.model(inputs) - targets).square().mean()
                if l1_weight:
                    loss = loss + l1_weight * sum(
                        parameter.abs().sum() for parameter in self.model.parameters()
                    )
                loss.backward()
                optimizer.step()
            self._updates += 1
            if epoch != 0 and (epoch + 1) % record_every != 0 and epoch + 1 != epochs:
                continue
            improved = self._record(stage)
            if log_every and (epoch == 0 or (epoch + 1) % log_every == 0):
                print(
                    f"{stage} {epoch + 1}: validation MSE={self._history[-1]['validation_mse']:.3e}",
                    flush=True,
                )
            if early_stopping is not None:
                if improved:
                    bad_epochs = 0
                elif epoch >= early_stopping.get("grace", 0):
                    bad_epochs += 1
                if bad_epochs > early_stopping["patience"]:
                    break

    def fit_lbfgs(self, optimizer, steps, log_every=None):
        """Run full-batch L-BFGS outer steps; its settings belong to the optimizer."""

        def closure():
            optimizer.zero_grad()
            loss = (self.model(self.train_x) - self.train_targets).square().mean()
            loss.backward()
            return loss

        for step in range(steps):
            self.model.train()
            optimizer.step(closure)
            self._updates += 1
            self._record("lbfgs")
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"lbfgs {step + 1}: validation MSE={self._history[-1]['validation_mse']:.3e}",
                    flush=True,
                )

    def restore_best(self):
        """Restore the selected checkpoint after completing all optimizer stages."""
        if self._best_state is None:
            raise ValueError("No validation checkpoint has been selected.")
        self.model.load_state_dict(self._best_state)
        self.model.eval()
