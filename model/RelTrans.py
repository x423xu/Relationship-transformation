import torch
from typing import Dict
import torch.nn as nn
from lib.synsin.models.z_buffermodel import ZbufferModelPts

# from ..lib.synsin.options.train_options import ArgumentParser
from .networks.RelSpa import RSModel, Reprojector


class Hook:
    def __init__(self, d=None) -> None:
        self._data = d

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d


class NewZbufferModelPts(ZbufferModelPts):
    def __init__(self, opt, model_path):
        super().__init__(opt)
        self.hook = Hook()
        """
        freeze unused parts
        """
        self._load_pretrain(model_path)
        self._freeze_module(self.encoder)
        self._freeze_module(self.projector)
        self._freeze_module(self.pts_regressor)

    def _load_pretrain(self,model_path):
        state_dict_new = {}
        for k,v in torch.load(model_path)["state_dict"].items():
            if 'model.module' in k:
                k = k.replace('model.module.','')
            if k not in self.state_dict().keys():
                continue
            state_dict_new[k] = v
        if len(state_dict_new.keys()) != len(self.state_dict().keys()):
            raise 'keys error'
        self.load_state_dict(state_dict_new)
        print('load from synsin')
        
    def _freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, batch, rel_embedding):
        input_img = batch["images"][0]
        K = batch["K"]
        K_inv = batch["Kinv"]
        input_RT = batch["P"][0]
        input_RTinv = batch["Pinv"][0]
        output_RT = batch["P"][1]
        output_RTinv = batch["Pinv"][1]

        fs = rel_embedding
        regressed_pts = (
            nn.Sigmoid()(self.pts_regressor(input_img))
            * (self.opt.max_z - self.opt.min_z)
            + self.opt.min_z
        )
        gen_fs = self.pts_transformer.forward_justpts(
            fs,
            regressed_pts,
            K,
            K_inv,
            input_RT,
            input_RTinv,
            output_RT,
            output_RTinv,
            self.hook,
        )
        pts3d = self.hook.data
        return gen_fs, pts3d


class RelTrans(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        """
        Relationship Spatializer
        """
        self.relationship_spatializer = RSModel()
        self.rel_embedding = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(), nn.Conv2d(256, 256, 3, 1, 1)
        )

        """
        z_buffer
        """
        MODEL_PATH = "lib/synsin/modelcheckpoints/realestate/zbufferpts.pth"
        opts = torch.load(MODEL_PATH)["opts"]
        self.z_buffer = NewZbufferModelPts(opts, model_path = MODEL_PATH)

        """
        reprojector
        """
        self.reprojector = Reprojector()

        """
        bbox_regressor
        """
        self.bbox_regressor = nn.Sequential(
            nn.Linear(8, 16), nn.LeakyReLU(), nn.Linear(16, 8)
        )




    def forward(self, batch: Dict):
        rel_features = batch["rel_features"][0]
        bbox = batch["bbox"][0]
        spatial_rel = self.relationship_spatializer(
            rel_features, bbox, [self.args.W, self.args.W]
        )
        rel_embedding = self.rel_embedding(spatial_rel)
        z_buffer, pts3d = self.z_buffer(batch, rel_embedding)
        fs_result = self.reprojector(z_buffer)
        return fs_result, pts3d
