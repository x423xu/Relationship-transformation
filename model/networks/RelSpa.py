"""
Relationship Spatialization
"""

import torch
import torch.nn as nn
import numpy as np


class RSModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def interpolate(
        self, rel_features, bbox, size
    ):  # (b,128,1024,1)->(b, 128, 240, 320),bbox(b,256,8)
        rel_features = rel_features.unsqueeze(-1)
        out = torch.zeros(
            [rel_features.shape[0], rel_features.shape[1], size[0], size[1]],
            dtype=rel_features.dtype,
            device=rel_features.device,
        )
        # print(rel_features.shape, out.shape, bbox.shape)
        interpolate_ = nn.functional.interpolate
        [w, w] = size
        sub_box = (bbox[:, :, :4]).int() // 2
        obj_box = (bbox[:, :, 4:]).int() // 2
        # sub_box = sub_box.cpu().numpy()
        # obj_box = obj_box.cpu().numpy()
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                sx1, sy1, sx2, sy2 = sub_box[i, j, :]
                sh, sw = sy2 - sy1, sx2 - sx1
                if sh < 5 or sw < 5:
                    continue
                sf = interpolate_(
                    rel_features[i, j, :, :].unsqueeze(0).unsqueeze(0), size=[sh, sw]
                )
                ox1, oy1, ox2, oy2 = obj_box[i, j, :]
                oh, ow = oy2 - oy1, ox2 - ox1
                if oh < 5 or ow < 5:
                    continue
                of = interpolate_(
                    rel_features[i, j, :, :].unsqueeze(0).unsqueeze(0), size=[oh, ow]
                )
                out[i, j, sy1:sy2, sx1:sx2] += sf.squeeze()
                out[i, j, oy1:oy2, ox1:ox2] += of.squeeze()
        return out

    def forward(self, rel_features, bbox, size):
        return self.interpolate(rel_features, bbox=bbox, size=size)


class Reprojector(nn.Module):
    def __init__(self, rel_dim=51) -> None:
        super().__init__()
        self.rel_dim = rel_dim
        self.embedding = nn.MultiheadAttention(rel_dim, 1)
        self.query_embedding = nn.Conv2d(256, 256, 3, 1, 1)
        self.key_embedding = nn.Conv2d(256, 256, 3, 1, 1)
        self.value_embedding = nn.Conv2d(256, 256, 3, 1, 1)
        self.act = nn.Softmax(dim=-1)

    def forward(self, fs):
        query = self.query_embedding(fs)
        key = self.key_embedding(fs)
        value = self.value_embedding(fs)
        query = nn.functional.interpolate(query, [self.rel_dim, 1]).squeeze()
        key = nn.functional.interpolate(key, [self.rel_dim, 1]).squeeze()
        value = nn.functional.interpolate(value, [self.rel_dim, 1]).squeeze()
        fs_reprojection, _ = self.embedding(query, key, value)
        fs_reprojection = self.act(fs_reprojection)
        return fs_reprojection
