"""Quick verification of NDeF-DIC module imports and basic functionality."""
import sys
sys.path.insert(0, '.')

import torch
print(f"[INFO] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# Config
from ndef_dic.config import NDeFDICConfig, SDFConfig, IntensityConfig, DeformationConfig, AppearanceConfig, TrainingConfig
print('config OK')

# Positional Encoding
from ndef_dic.encoding import PositionalEncoding
pe = PositionalEncoding(n_freqs=6, input_dim=3)
x = torch.randn(4, 3)
y = pe(x)
expected = 3 + 2 * 3 * 6
assert y.shape == (4, expected), f'Expected (4, {expected}), got {y.shape}'
print(f'encoding OK: in={x.shape} -> out={y.shape} (expected {expected})')

# Networks
from ndef_dic.networks import SDFNetwork, IntensityField, DeformationField, AppearanceEmbedding

sdf = SDFNetwork(SDFConfig())
s = sdf(torch.randn(4, 3))
print(f'SDF OK: {s.shape}')

intensity = IntensityField(IntensityConfig())
g = intensity(torch.randn(4, 3))
print(f'Intensity OK: {g.shape}')

deform = DeformationField(DeformationConfig())
d = deform(torch.randn(4, 3), torch.rand(4, 1))
print(f'Deformation OK: {d.shape}')

app = AppearanceEmbedding(AppearanceConfig(), n_cameras=4)
scale, bias = app(torch.tensor([0, 1, 2, 3]))
print(f'Appearance OK: scale={scale.shape}, bias={bias.shape}')
corrected = app.correct(torch.randn(4, 1), torch.tensor([0, 1, 2, 3]))
print(f'Appearance correct OK: {corrected.shape}')

# Renderer (basic check without full pipeline)
from ndef_dic.renderer import generate_rays, sphere_trace, project_points
K = torch.tensor([[2000., 0, 960.], [0, 2000., 600.], [0, 0, 1.]])
R = torch.eye(3)
t = torch.zeros(3)
rays_o, rays_d = generate_rays(K, R, t, 64, 64)
print(f'Ray generation OK: rays_o={rays_o.shape}, rays_d={rays_d.shape}')

# Sphere trace test
x_hit, hit_mask, t_vals = sphere_trace(rays_o[:4], rays_d[:4], sdf)
print(f'Sphere trace OK: x_hit={x_hit.shape}, hit_rate={hit_mask.float().mean():.2f}')

# Projection test
pts = torch.randn(8, 3)
uv, depth = project_points(pts, K, R, t)
print(f'Projection OK: uv={uv.shape}, depth={depth.shape}')

# Losses
from ndef_dic.losses import mse_photo_loss, eikonal_loss
l_eik = eikonal_loss(sdf, torch.randn(32, 3))
print(f'Eikonal loss OK: {l_eik.item():.4f}')

print()
print('='*60)
print('All modules verified successfully!')
print('='*60)
