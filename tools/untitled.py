import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
device = 'cuda'
# pointcloud = np.load('pointcloud.npz')
image = cv2.imread('/home/xxy/Documents/data/RealEstate10K/videos/test/0c0f298ace7c875b/155388567.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, [256,256])
# w,h = img.shape
# img = img.view(-1,3)
# depth = cv2.imread('depth.png')
# x = np.linspace(0,w)
verts = torch.Tensor(np.load('p1.npy', allow_pickle=True)).squeeze().to(device)
verts[:,1] = - verts[:,1]
verts[:,0] = - verts[:,0]
verts[:,2] = 1*(verts[:,2]-verts[:,2].min())/ (verts[:,2].max()-verts[:,2].min())
# verts = 10*(verts-verts.min())/ (verts.max()-verts.min())+0.1
rgb = torch.Tensor(img).view(-1,3).to(device)
rgb = torch.hstack([rgb, torch.ones(rgb.shape[0], 1).to(device)])/255
# verts = torch.Tensor(pointcloud['verts']).to(device)
        
# rgb = torch.Tensor(pointcloud['rgb']).to(device)

point_cloud = Pointclouds(points=[verts], features=[rgb])

R, T = look_at_view_transform(20, 10, 0)
cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
raster_settings = PointsRasterizationSettings(
    image_size=256, 
    radius = 0.03,
    points_per_pixel = 4
)
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = PointsRenderer(
    rasterizer=rasterizer,
    compositor=AlphaCompositor()
)

images = renderer(point_cloud)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()
# renderer = PointsRenderer(
#     rasterizer=rasterizer,
#     # Pass in background_color to the alpha compositor, setting the background color 
#     # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
#     compositor=AlphaCompositor(background_color=(0, 0, 1))
# )
# images = renderer(point_cloud)

# plt.figure(figsize=(10, 10))
# plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.axis("off")
# plt.show()
