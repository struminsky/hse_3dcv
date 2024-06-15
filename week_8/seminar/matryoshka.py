import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import normalize

### SDF

def sdf(points):
    def sdf_sphere(points, center, radius):
        return torch.norm(points - center, dim=-1) - radius
    def smooth_minimum(a, b, k):
        h = (k - (a - b).abs()).clamp(0) / k
        return torch.minimum(a, b) - h * h * k / 4
    # matryoshka
    matryoshka = smooth_minimum(
        sdf_sphere(points, torch.as_tensor([0.0, 0.0, 1.5]).to(points), 0.75),  # head
        sdf_sphere(points, torch.as_tensor([0.0, 0.0, 0.5]).to(points), 0.9),  # body
        k=0.3
    )
    # background
    background = smooth_minimum(
        8. - torch.norm(points[..., :2], dim=-1),
        points[..., 2],
        k=4.0)
    return torch.minimum(background, matryoshka)

### Colors

class Texture(nn.Module):
    def __init__(self, resolution=512, texture_grid=None, background_color=None, train_background=True):
        super().__init__()
        if texture_grid is None:
            grid_shape = [1, 3, resolution, resolution]
            texture_grid = torch.full(grid_shape, 0.8)
        else:
            texture_grid = F.interpolate(texture_grid, (resolution, resolution))
        self.matryoshka_grid = nn.Parameter(texture_grid)
        if background_color is None:
            background_color = torch.as_tensor([0.6, 0.5, 0.4])
        self.background_color = nn.Parameter(background_color, requires_grad=train_background)
        self.fading = 0.1
        self.fading_border_width = int(self.fading * resolution)
    
    def forward(self, points):
        batch_shape = points.shape[:-1]
        
        x = points[..., :1]
        y = points[..., 1:2]
        z = points[..., 2:]
        
        r = torch.norm(points[..., :2], dim=-1, keepdim=True)
        phi = torch.acos(x / r)
        #phi = torch.where(y >= 0, phi, -phi) / torch.pi
        phi = 2 * torch.where(y >= 0, phi, -phi) / torch.pi
        z_normalized = (z - 1.1) / 1.2
        
        matryoshka_front = F.grid_sample(
            self.matryoshka_grid,
            torch.cat([phi, z_normalized], dim=-1).view(1, 1, -1, 2),
            align_corners=False,
            padding_mode='zeros'
        ).view(3, -1).T.view(*points.shape)
        matryoshka_back = self.matryoshka_grid[0, :, -self.fading_border_width:, :]
        matryoshka_back = matryoshka_back.view(3, -1).mean(1)
        w = ((phi.abs() - (1 - self.fading)) / self.fading).clamp(0, 1)
        matryoshka_colors = (1 - w) * matryoshka_front + w * matryoshka_back
        #matryoshka_colors = matryoshka_back
        return torch.where(torch.logical_and(r < 1., z > 1e-2),
                           matryoshka_colors,
                           self.background_color)

def calculate_normals(points):
    points.requires_grad_()
    return normalize(torch.autograd.grad([sdf(points).sum()], [points])[0], dim=-1)

def calculate_soft_shadow(rays):
    t = 1e-3 * torch.ones(*rays['origins'].shape[:-1],
                          device=rays['origins'].device)
    dist = torch.ones_like(t)
    for _ in range(32):
        points = rays['origins'] + t[..., None] * rays['directions']
        sdf_values = sdf(points)
        t = torch.where(sdf_values < 1e-4, t, t + sdf_values)
        dist = torch.minimum(dist, 4. * dist / t)
    return 1. - dist.unsqueeze(-1)

### Renderer

def get_rays(focal_length, resolution, device):
    ray_origins = torch.zeros(resolution, resolution, device=device)
    image_coordinates = torch.stack(torch.meshgrid([
        torch.linspace(-1, 1, resolution, device=device),
        torch.linspace(-1, 1, resolution, device=device)
    ], indexing='xy'), dim=-1)
    ray_directions = normalize(torch.cat([
        image_coordinates,
        focal_length * torch.ones_like(image_coordinates[..., :1])
    ], dim=-1), dim=-1)
    return {'origins': ray_origins,
            'directions': ray_directions}

def translate_to_world_coordinates(rays, camera_origins, camera_directions):
    camera_zs = normalize(camera_directions)
    world_zs = torch.broadcast_to(torch.eye(3, device=rays['origins'].device)[2], camera_zs.shape)
    camera_xs = normalize(torch.cross(camera_zs, world_zs), dim=-1)
    camera_ys = normalize(torch.cross(camera_zs, camera_xs), dim=-1)
    # rotation matrices shape is batch x 3 x 3
    rotation_matrices = torch.stack([camera_xs, camera_ys, camera_zs], dim=-1)

    # rays have shape h x w x 3
    world_directions = torch.einsum(
        'bij,hwj->bhwi',
        rotation_matrices,
        rays['directions']
    )
    world_origins = torch.broadcast_to(
        camera_origins.view(-1, 1, 1, 3),
        world_directions.shape
    )

    return {'origins': world_origins,
            'directions': world_directions}

def march_rays(rays):
    t = torch.zeros(*rays['origins'].shape[:-1],
                    device=rays['origins'].device)
    for _ in range(32):  # yes, its hardcoded
        points = rays['origins'] + t[..., None] * rays['directions']
        sdf_values = sdf(points)
        t = torch.where(sdf_values < 1e-4, t, t + sdf_values)
    return t

def render(camera_origins,
           camera_directions,
           get_pixel_colors,
           texture=None,
           light_source=None,
           focal_length=1.8,
           resolution=128,
           **kwargs):
    with torch.no_grad():
        rays = get_rays(focal_length, resolution, device=camera_origins.device)
        rays_world = translate_to_world_coordinates(
            rays,
            camera_origins,
            camera_directions
        )
        ts = march_rays(rays_world)
        surface_points = rays_world['origins'] + ts[..., None] * rays_world['directions']
    # compute colors
    image = get_pixel_colors(surface_points, texture, light_source)
    return image
