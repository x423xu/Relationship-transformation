import torch
import torch.nn as nn
from RelSpa import RSModel
from .z_buffer_manipulator import PtsManipulator
from .architectures import (
    Unet,
    UNetDecoder64,
    UNetEncoder64,
)


'''
given 
'''
class ZbufferModelPts(nn.Module):
    def __init__(self, args):
        super().__init__()

        '''
        Relationship Spatializer
        '''
        self.relationship_spatializer = RSModel()

        '''
        encoder: 
        '''
        self.encoder = UNetEncoder64(channels_in=3, channels_out=64)

        '''
        Regress 3D points
        '''
        self.pts_regressor = Unet(channels_in=3, channels_out=1)
        self.pts_transformer = PtsManipulator(args.W, opt=args)

        '''
        decoder
        '''
        self.decoder = UNetDecoder64(channels_in=64, channels_out=3)
        self.reprojector = nn.Identity()
    def forward(self, rgb, rel_features, bbox):
        spatial_rel = self.relationship_spatializer(rel_features, bbox)
        rel_embedding = self.encoder(spatial_rel)
        regressed_pts = (
                    nn.Sigmoid()(self.pts_regressor(rgb))
                    * (self.opt.max_z - self.opt.min_z)
                    + self.opt.min_z
                )
