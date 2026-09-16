"""Klein manifold."""

import torch
from torch import Tensor

from flow_klein.geometry.base import Manifold
from flow_klein.geometry.math_utils import arcosh, tanh, artanh

class Klein(Manifold):
    """
    Klein manifold class.

    We use the following convention: x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Klein ball radius.
    """

    def __init__(self):
        super(Klein, self).__init__()
        self.name = 'Klein'
        self.min_norm = 1e-15
        self.eps = {
            torch.float32: 4e-3,
            torch.float64: 1e-5,
            torch.bfloat16: 1e-2,
            torch.float16: 2e-2,
        }
        self.c = 1.0

    # --------------------------------------------------
    # Helper: λ(x) = 1 / sqrt(1 - c||x||²)
    # --------------------------------------------------

    def _lambda_x(self, x: Tensor, c: float) -> Tensor:          # (...,1)
        x_sqnorm = (x * x).sum(-1, keepdim=True)                 # (...,1)
        denom = (1.0 - c * x_sqnorm).clamp_min(self.min_norm)
        return 1.0 / torch.sqrt(denom)
    
    # --------------------------------------------------
    # Klein metric tensor  G(x)  (batch, dim, dim)
    # Not yet implemented in the toy code
    # --------------------------------------------------
    def klein_metric_tensor(self, x: Tensor, c: float) -> Tensor:
        """
        Compute the Klein metric tensor in matrix form G(x) at point x.
        G(x) =  I/(1-||x||²)  +  x xᵀ /(1-||x||²)²
        """
        norm_sq = (x * x).sum(-1, keepdim=True)                  # (...,1)
        dim     = x.size(-1)
        I       = torch.eye(dim, device=x.device, dtype=x.dtype)
        I       = I.expand(*x.shape[:-1], dim, dim)              # (...,d,d)
        outer   = x.unsqueeze(-1) @ x.unsqueeze(-2)              # (...,d,d)
        denom  = (1.0 - norm_sq).clamp_min(self.min_norm)       # (...,1)
        G = I / denom.unsqueeze(-1) + outer / (denom**2).unsqueeze(-1)
        return G 
    
    # --------------------------------------------------
    # Klein norm  ‖v‖_K  (...,)
    # --------------------------------------------------
    def klein_norm(self, x: Tensor, v: Tensor, c: float) -> Tensor:
        """
        Compute the norm of tangent vector v at point x.
        ‖v‖_K = sqrt( vᵀ G(x) v )
        """
        G   = self.klein_metric_tensor(x, c)                     # (...,d,d)
        v_  = v.unsqueeze(-1)                                    # (...,d,1)
        vTGv = (v_.transpose(-2, -1) @ (G @ v_)).squeeze(-1).squeeze(-1)
        return vTGv.clamp_min(self.min_norm).sqrt()              # (...,) 
    

    # ------------------------------------------------------------------
    # d_K(x,y)²   (...,)
    # ------------------------------------------------------------------
    def sqdist(self, x: Tensor, y: Tensor, c: float) -> Tensor:
        """
        Compute the geodesic distance between two points x and y in the Klein model.
        """
        xy       = (x * y).sum(-1)                                # (...,)
        x2, y2   = (x * x).sum(-1), (y * y).sum(-1)               # (...,)
        num      = 1.0 - xy
        den      = ((1.0 - x2) * (1.0 - y2)).clamp_min(self.min_norm).sqrt()
        return arcosh(num / den).pow(2)                           # (...,)
    
    def egrad2rgrad(self, x: Tensor, egrad: Tensor, c: float) -> Tensor:
        G_inv = torch.linalg.inv(self.klein_metric_tensor(x, c))  
        return (G_inv @ egrad.unsqueeze(-1)).squeeze(-1)  

    
    # --------------------------------------------------
    # Point projection
    # --------------------------------------------------
    def proj(self, x: Tensor, c: float) -> Tensor:
        if not torch.is_tensor(x):
            return x

        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        eps = self.eps.get(x.dtype, 4e-3)
        maxnorm = (1 - eps) / (c**0.5)
        cond = norm > maxnorm
        projected = torch.where(cond, x / norm * maxnorm, x)
        return torch.nan_to_num(projected, nan=0.0)

    # --------------------------------------------------
    # Tangent‑vector projection
    # --------------------------------------------------
    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u
    
    # --------------------------------------------------
    # Exponential map  exp_x(v)  -> (...,dim)
    # --------------------------------------------------
    def expmap(self, x: Tensor, v: Tensor, c: float) -> Tensor:
        nv   = self.klein_norm(x, v, c).unsqueeze(-1)             # (...,1)
        lamx = self._lambda_x(x, c)                               # (...,1)
        xv   = (x * v).sum(-1, keepdim=True)                      # (...,1)
        
        sinh, cosh = torch.sinh(nv), torch.cosh(nv)
        num = sinh * v / nv.clamp_min(self.min_norm)              # (...,d)
        den = cosh + (lamx**2) * xv * sinh / nv.clamp_min(self.min_norm)  # (...,1)
        y   = x + num / den.clamp_min(self.min_norm)              # (...,d)
        return y             
    
    def expmap0(self, u: Tensor, c: float) -> Tensor:
        u_norm  = u.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        gamma_1 = tanh(u_norm) * u / u_norm
        return gamma_1
    
    # ------------------------------------------------------------------
    # logmap_x(y)   (...,d)
    # ------------------------------------------------------------------
    def logmap(self, x: Tensor, y: Tensor, c: float) -> Tensor:
        """
        Logarithmic map on the Klein model.
        Maps point y back to the tangent space at x.
        """
        diff       = y - x                                             # (..., d)
        norm_diff  = self.klein_norm(x, diff, c).unsqueeze(-1)         # (..., 1)
        sqdist_xy  = self.sqdist(x, y, c).clamp_min(self.min_norm)     # (...,)
        dist_xy    = torch.sqrt(sqdist_xy).unsqueeze(-1)               # (..., 1)
        v = dist_xy * diff / norm_diff                                 # (..., d)

        # when x ≈ y
        mask = norm_diff.squeeze(-1) < self.min_norm
        if mask.any():
            v[mask] = 0.0
        return v  

    def logmap0(self, p, c):
        p_norm = torch.clamp_min(p.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        scale = arcosh(self._lambda_x(p, c)) / p_norm
        return scale * p

    # ------------------------------------------------------------------
    # Mobius addition and matrix-vector multiplication
    # ------------------------------------------------------------------
    def mobius_add(self, x: Tensor, y: Tensor, c: float, dim: int = -1) -> Tensor:
        """
        We adopt the same function name as the counterparts in the Poincare ball and hyperboloid models for ease of use. 
        This function in fact implements the Einstein addition of two vectors in the Klein model.
        """
        x_proj = self.proj(x, c)
        y_proj = self.proj(y, c)
        xy = (x_proj * y_proj).sum(dim=dim, keepdim=True)
        x2 = (x_proj * x_proj).sum(dim=dim, keepdim=True)
        gamma_x = 1.0 / (1.0 - x2).clamp_min(self.min_norm).sqrt()
        denom = (1.0 + xy).clamp_min(self.min_norm)
        res = (x_proj + y_proj / gamma_x + gamma_x / (1.0 + gamma_x) * xy * x_proj) / denom
        res = self.proj(res, c)
        return torch.nan_to_num(res, nan=0.0)
    
    def mobius_matvec(self, m: Tensor, x: Tensor, c: float = 1.0) -> Tensor:
        """
        We adopt the same function name as the counterparts in the Poincare ball and hyperboloid models for ease of use.
        This function in fact implements the Einstein matrix-vector multiplication in the Klein model.
        """

        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch.Tensor")

        x_proj = self.proj(x, c)
        x_norm = x_proj.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)      # (...,1)
        mx = x_proj @ m.transpose(-1, -2)                                        # (...,d)
        mx_norm = mx.norm(dim=-1, keepdim=True)

        sqrt_term = (1.0 - x_norm ** 2).clamp_min(self.min_norm).sqrt()
        arg = x_norm / (1.0 + sqrt_term)
        arg = arg.clamp(max=1.0 - 1e-6)

        theta = artanh(arg)
        kappa = 2.0 * theta * mx_norm.clamp_min(self.min_norm) / x_norm
        scale = tanh(kappa) / mx_norm.clamp_min(self.min_norm)

        res_c = scale * mx
        res_c = torch.where(mx_norm <= self.min_norm, torch.zeros_like(res_c), res_c)
        res_c = self.proj(res_c, c)
        return torch.nan_to_num(res_c, nan=0.0)
    
    
    # ------------------------------------------------------------
    # Parallel transport through the Lorentz isometry
    # ------------------------------------------------------------
    @staticmethod
    def _lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
        """Lorentzian inner product with time-like coordinate first."""
        return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(-1, keepdim=True)

    def ptransp(self, x: Tensor, y: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
        """
        Analytic parallel transport from T_xK to T_yK along their connecting
        geodesic, evaluated through the unit-curvature Lorentz isometry.

        Args:
            x: Source point on Klein manifold (..., dim)
            y: Target point on Klein manifold (..., dim)
            v: Tangent vector at x to be transported (..., dim)
            c: Curvature parameter
        Returns:
            Transported vector in T_yK (..., dim)
        """
        if float(c) != 1.0:
            raise NotImplementedError(
                "Lorentz parallel transport currently supports only unit curvature c=1"
            )

        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)

        point_x = torch.cat((lambda_x, lambda_x * x), dim=-1)
        point_y = torch.cat((lambda_y, lambda_y * y), dim=-1)

        x_dot_v = (x * v).sum(-1, keepdim=True)
        tangent_v = torch.cat(
            (
                lambda_x.pow(3) * x_dot_v,
                lambda_x * v + lambda_x.pow(3) * x_dot_v * x,
            ),
            dim=-1,
        )

        coefficient = self._lorentz_inner(point_y, tangent_v)
        coefficient = coefficient / (
            1.0 - self._lorentz_inner(point_x, point_y)
        ).clamp_min(self.min_norm)
        transported = tangent_v + coefficient * (point_x + point_y)

        time = transported[..., :1]
        spatial = transported[..., 1:]
        return (spatial - time * y) / lambda_y

    def ptransp0(self, x: Tensor, u: Tensor, c: float = 1.0) -> Tensor:
        """Parallel transport of u from the origin to T_xK."""
        return self.ptransp(torch.zeros_like(x), x, u, c)

    def _ptransp_to_origin(self, x: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
        """Parallel transport of v from T_xK to the origin."""
        return self.ptransp(x, torch.zeros_like(x), v, c)

    # ------------------------------------------------------------
    # Legacy parallel transport retained for result reproduction
    # ------------------------------------------------------------
    def ptransp0_legacy(self, x: Tensor, u: Tensor, c: float = 1.0) -> Tensor:
        """Legacy origin-to-x transport formula."""
        x_norm_sq = (x * x).sum(-1, keepdim=True)                           # (...,1)
        x_dot_u   = (x * u).sum(-1, keepdim=True)                           # (...,1)
        sqrt_term = (1.0 - x_norm_sq).clamp_min(self.min_norm).sqrt()       # (...,1)
        coeff     = x_dot_u * (sqrt_term - 2.0) / (1.0 - sqrt_term).clamp_min(self.min_norm)  # (...,1)
        return coeff * x + sqrt_term * u                                    # (...,d)

    def ptransp_legacy(self, x: Tensor, y: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
        """Legacy transport via the origin, retained for reproducibility."""
        v_at_origin = self._ptransp_to_origin_legacy(x, v, c)
        return self.ptransp0_legacy(y, v_at_origin, c)

    def _ptransp_to_origin_legacy(self, x: Tensor, v: Tensor, c: float = 1.0) -> Tensor:
        """
        Legacy x-to-origin formula used by :meth:`ptransp_legacy`.
        """
        x_norm_sq = (x * x).sum(-1, keepdim=True)                           # (...,1)
        sqrt_term = (1.0 - x_norm_sq).clamp_min(self.min_norm).sqrt()       # (...,1)
        v_dot_x = (v * x).sum(-1, keepdim=True)                             # (...,1)
        
        # Inverse transport formula
        coeff = v_dot_x / (1.0 + sqrt_term).clamp_min(self.min_norm)        # (...,1)
        v_at_origin = sqrt_term * (v - coeff * x)                           # (...,d)
        
        return v_at_origin
    
    # ------------------------------------------------------------
    # Calculating x_t and dxt
    # ------------------------------------------------------------
    class KleinGeodesic:
        """Utility class to compute geodesic interpolation X_t and velocity dX_t on Klein model."""
        def __init__(self, min_norm: float = 1e-15):
            self.min_norm = min_norm

        def dot(self, x, y):
            return (x * y).sum(dim=-1, keepdim=True)

        def sqnorm(self, x):
            return (x * x).sum(dim=-1, keepdim=True)

        def dist(self, x, y):
            num = 1 - self.dot(x, y)
            den = ((1 - self.sqnorm(x)) * (1 - self.sqnorm(y))).clamp_min(self.min_norm).sqrt()
            return torch.acosh(num / den)

        def unit_vec_at_x0(self, x, y):
            num = (y - x) * (1 - self.sqnorm(x))
            den = ((1 - self.sqnorm(x)) * self.sqnorm(y - x) + self.dot(y - x, x) ** 2).sqrt().clamp_min(self.min_norm)
            return num / den

        @staticmethod
        def coth(z):
            return 1.0 / torch.tanh(z)

        def xt(self, x0, x1, t):
            """Geodesic interpolation X_t on Klein."""
            uv   = self.unit_vec_at_x0(x0, x1)
            frac = self.dot(x0, uv) / (1 - self.sqnorm(x0))                     # (B,1)
            td   = t.view(-1, 1) * self.dist(x0, x1)                            # (B,1)
            denom = self.coth(td) + frac                                        # (B,1)
            return x0 + uv / denom.clamp_min(self.min_norm)                          # (B,d)

        def dxt(self, x0, x1, t):
            """Velocity field dX_t along the geodesic."""
            uv   = self.unit_vec_at_x0(x0, x1)
            frac = self.dot(x0, uv) / (1 - self.sqnorm(x0))                     # (B,1)
            dist = self.dist(x0, x1)                                            # (B,1)
            td   = t.view(-1, 1) * dist                                         # (B,1)
            coth_td = self.coth(td)
            denom   = (coth_td + frac) ** 2
            numer   = dist * uv * (coth_td ** 2 - 1)
            return numer / denom.clamp_min(self.min_norm)

    geo = KleinGeodesic(min_norm=1e-15)
