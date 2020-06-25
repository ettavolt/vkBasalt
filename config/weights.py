import torch
from torch.nn import functional as nnF
from io import open
from struct import Struct


# weights = [
#     torch.ones((3,)),
#     torch.zeros((3, 3, 3)),
# ]
# weights[1][1, 1] = -weights[0]

weights = [
    torch.zeros((32, 3, 3, 3)),
    torch.zeros((32,)),
    torch.zeros((32, 3, 3, 3)),
    torch.zeros((3,)),
]
weights[1][0:12] = 1.0
for x in range(0, 2):
    for y in range(0, 2):
        for i in range(0, 3):
            weights[0][x * 6 + y * 3 + i, i, x+1, y+1] = -1.0
            weights[2][x * 6 + y * 3 + i, i, x+1, y+1] = 1.0

batch = torch.linspace(0, 191, steps=192).reshape((1, 8, 8, 3)).transpose(2, 3).transpose(1, 2).true_divide(100)
out = nnF.conv2d(
    batch,
    weights[0],
    bias=weights[1],
    stride=2,
    padding=1
)
out = nnF.conv_transpose2d(
    out,
    weights[2],
    bias=weights[3],
    stride=2,
    padding=1,
    output_padding=1
)
print(batch.add(out).mean().item())

ff = Struct('f')
with open('./weights.bin', 'wb') as f:
    for t in weights:
        for num in t.flatten():
            f.write(ff.pack(num))
