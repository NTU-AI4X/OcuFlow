from __future__ import annotations

"""Small training helpers shared by the three-stage trainer and tests."""

from typing import Iterable, Optional

import torch


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    gradients = []
    device: Optional[torch.device] = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        device = grad.device
        gradients.append(torch.linalg.vector_norm(grad.float(), ord=2))
    if not gradients:
        return torch.zeros((), device=device or torch.device("cpu"))
    return torch.linalg.vector_norm(torch.stack(gradients), ord=2)


def clip_or_measure_gradients(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
) -> torch.Tensor:
    """Return the pre-clip norm; max_norm=0 explicitly disables clipping."""
    parameters = list(parameters)
    if float(max_norm) > 0.0:
        return torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    return gradient_norm(parameters)


def backward_renderer_losses(losses, proxy, *, scale: float, propagate_to_pmf: bool):
    """Update the gate with total loss but return only data-loss dL/dx0 for PMF."""
    scaled_total = losses.total * float(scale)
    if not torch.isfinite(scaled_total):
        raise FloatingPointError("nonfinite renderer loss")
    proxy_gradient = None
    if propagate_to_pmf:
        if proxy is None or not proxy.requires_grad:
            raise ValueError("propagate_to_pmf requires a differentiable x0 proxy")
        scaled_data = losses.data * float(scale)
        (proxy_gradient,) = torch.autograd.grad(
            scaled_data, proxy, retain_graph=True, create_graph=False, allow_unused=False
        )
        proxy_gradient = proxy_gradient.detach()
    scaled_total.backward()
    return proxy_gradient
