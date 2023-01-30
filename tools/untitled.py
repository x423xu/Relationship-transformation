import torch

W=256

xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)
print(xs, ys)
xyzs = torch.cat(
    (xs, -ys, -torch.ones(xs.size()), torch.ones(xs.size())), 1
).view(1, 4, -1)

print(xyzs, xyzs.shape)