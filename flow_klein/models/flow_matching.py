"""
Klein Geodesic Flow Matching Model with conditional DiT velocity prediction.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from flow_klein.geometry.klein import Klein
from flow_klein.models.dit import ConditionalDiT


class KleinFlowMatching(nn.Module):
    """
    Conditional flow matching on the Klein manifold.

    The flow is trained in a standardized tangent space and mapped back to the
    original latent geometry for decoding.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_timesteps: int = 200,
        base_std: float = 1.0,
        use_simple_dit: bool = False,
        dropout: float = 0.0,
        flow_cond_dropout: float = 0.1,
        use_cond_guidance: bool = True,
        guidance_scale: float = 1.2,
        device: str = 'cuda'
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_timesteps = num_timesteps
        self.base_std = base_std
        self.device = device
        self.flow_cond_dropout = flow_cond_dropout
        self.use_cond_guidance = use_cond_guidance
        self.guidance_scale = guidance_scale
        self.klein = Klein()

        input_dim = dim * 2 + 2
        self.dit = ConditionalDiT(
            input_dim=input_dim,
            output_dim=dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.register_buffer('data_mean', torch.zeros(1, dim))
        self.register_buffer('data_std', torch.ones(1, dim))

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.data_mean.copy_(mean.view(1, -1))
        self.data_std.copy_(std.view(1, -1).clamp_min(1e-4))

    def normalize_to_model_space(self, x_klein: torch.Tensor) -> torch.Tensor:
        tangent = self.klein.logmap0(self.klein.proj(x_klein, c=1.0), c=1.0)
        normalized = (tangent - self.data_mean) / self.data_std
        normalized_klein = self.klein.expmap0(normalized, c=1.0)
        return self.klein.proj(normalized_klein, c=1.0)

    def denormalize_from_model_space(self, x_model: torch.Tensor) -> torch.Tensor:
        tangent = self.klein.logmap0(self.klein.proj(x_model, c=1.0), c=1.0)
        restored = tangent * self.data_std + self.data_mean
        restored_klein = self.klein.expmap0(restored, c=1.0)
        return self.klein.proj(restored_klein, c=1.0)

    def sample_base(self, batch_size: int) -> torch.Tensor:
        tangent_samples = torch.randn(batch_size, self.dim, device=self.device) * self.base_std
        klein_samples = self.klein.expmap0(tangent_samples, c=1.0)
        return self.klein.proj(klein_samples, c=1.0)

    def _geometry_features(self, x: torch.Tensor) -> torch.Tensor:
        tangent = self.klein.logmap0(x, c=1.0)
        radius = x.norm(dim=-1, keepdim=True)
        lambda_x = self.klein._lambda_x(x, c=1.0)
        return torch.cat([x, tangent, radius, lambda_x], dim=-1)

    def _drop_condition(self, cond_code: torch.Tensor) -> torch.Tensor:
        if not self.training or self.flow_cond_dropout <= 0:
            return cond_code
        drop_mask = (torch.rand(cond_code.shape[0], 1, device=cond_code.device) < self.flow_cond_dropout).float()
        return cond_code * (1.0 - drop_mask)

    def _predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_code: torch.Tensor,
        guidance_scale: float = None
    ) -> torch.Tensor:
        features = self._geometry_features(x)
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale

        if self.training or not self.use_cond_guidance or guidance_scale == 1.0:
            return self.dit(features, t, cond_code)

        pred_cond = self.dit(features, t, cond_code)
        pred_uncond = self.dit(features, t, torch.zeros_like(cond_code))
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    def forward(self, x1: torch.Tensor, cond_code: torch.Tensor) -> torch.Tensor:
        batch_size = x1.shape[0]
        device = x1.device

        x1 = self.normalize_to_model_space(x1)
        x1 = self.klein.proj(x1, c=1.0)
        x0 = self.sample_base(batch_size)

        eps = 1e-4
        t = torch.rand(batch_size, device=device) * (1 - 2 * eps) + eps
        x_t = self.klein.geo.xt(x0, x1, t)
        x_t = self.klein.proj(x_t, c=1.0)
        target_velocity = self.klein.geo.dxt(x0, x1, t)

        cond_code = self._drop_condition(cond_code)
        pred_velocity = self._predict_velocity(x_t, t, cond_code)
        return self._klein_mse_loss(pred_velocity, target_velocity, x_t)

    def _klein_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        diff = pred - target
        klein_norm_sq = self.klein.klein_norm(x, diff, c=1.0).pow(2)
        return klein_norm_sq.mean()

    def loss_fn(self, x1: torch.Tensor, cond_code: torch.Tensor) -> torch.Tensor:
        return self.forward(x1, cond_code)

    @torch.no_grad()
    def sample(
        self,
        cond_code: torch.Tensor,
        steps: int = None,
        return_trajectory: bool = False,
        guidance_scale: float = None
    ) -> torch.Tensor:
        steps = steps or self.num_timesteps
        dt = 1.0 / steps
        batch_size = cond_code.shape[0]

        current = self.sample_base(batch_size)
        trajectory = [self.denormalize_from_model_space(current)] if return_trajectory else None

        for step in tqdm(range(steps), desc="Klein Flow Sampling"):
            t_value = step / steps
            t_tensor = torch.full((batch_size,), t_value, device=self.device)
            velocity = self._predict_velocity(current, t_tensor, cond_code, guidance_scale=guidance_scale)
            current = self.klein.expmap(current, velocity * dt, c=1.0)
            current = self.klein.proj(current, c=1.0)

            if return_trajectory:
                trajectory.append(self.denormalize_from_model_space(current))

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return self.denormalize_from_model_space(current)

    @torch.no_grad()
    def sample_heun(
        self,
        cond_code: torch.Tensor,
        steps: int = None,
        guidance_scale: float = None
    ) -> torch.Tensor:
        steps = steps or self.num_timesteps
        dt = 1.0 / steps
        batch_size = cond_code.shape[0]

        current = self.sample_base(batch_size)

        for step in tqdm(range(steps), desc="Klein Flow Sampling (Heun)"):
            t_value = step / steps
            t_tensor = torch.full((batch_size,), t_value, device=self.device)
            t_next = torch.full((batch_size,), (step + 1) / steps, device=self.device)

            v1 = self._predict_velocity(current, t_tensor, cond_code, guidance_scale=guidance_scale)
            x_euler = self.klein.expmap(current, v1 * dt, c=1.0)
            x_euler = self.klein.proj(x_euler, c=1.0)

            v2 = self._predict_velocity(x_euler, t_next, cond_code, guidance_scale=guidance_scale)
            v2_transported = self.klein.ptransp(x_euler, current, v2, c=1.0)
            v_avg = (v1 + v2_transported) / 2.0

            current = self.klein.expmap(current, v_avg * dt, c=1.0)
            current = self.klein.proj(current, c=1.0)

        return self.denormalize_from_model_space(current)

    def to_tangent_space(self, x_klein: torch.Tensor) -> torch.Tensor:
        return self.klein.logmap0(x_klein, c=1.0)

    def from_tangent_space(self, x_tangent: torch.Tensor) -> torch.Tensor:
        x_klein = self.klein.expmap0(x_tangent, c=1.0)
        return self.klein.proj(x_klein, c=1.0)
