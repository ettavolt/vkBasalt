from math import gcd
from tqdm import tqdm

# NVidia offers prebuilds for GNU/Linux amd64,
# see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html

import nvidia.dali as dali
import torch
import torch.nn as nn
import av
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe = dali.pipeline.Pipeline(batch_size=1, num_threads=2, device_id=0)
with pipe:
    frames_src = dali.fn.video_reader(
        name="source",
        device="gpu",
        sequence_length=8,
        dtype=dali.types.FLOAT,
        filenames=["dragonguard.mkv"],
        normalized=True,
        skip_vfr_check=True,
        tensor_init_bytes=1920*1080*3*4,
    )
    pipe.set_outputs(frames_src)
pipe.build()
sampler = DALIGenericIterator(
    [pipe],
    ['frames'],
    reader_name="source",
    fill_last_batch=False,
)


class ResidualSeparableBlock(nn.Module):
    def __init__(self, features, kernel=3, padding=1, padding_mode='reflect', **kwargs):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                features,
                features,
                kernel,
                padding=padding,
                padding_mode=padding_mode,
                groups=features,
                **kwargs
            ),
            nn.Conv2d(features, features, 1, bias=False),
            nn.InstanceNorm2d(features, affine=True),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                features,
                features,
                kernel,
                padding=padding,
                padding_mode=padding_mode,
                groups=features,
                **kwargs
            ),
            nn.Conv2d(features, features, 1, bias=False),
            nn.InstanceNorm2d(features, affine=True),
        )

    def forward(self, x):
        return self.conv_2(self.conv_1(x)) + x


def conv_separable_block(in_features, out_features, kernel=3, stride=2, padding=1, padding_mode='reflect', **kwargs):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=gcd(in_features, out_features),
            **kwargs
        ),
        nn.Conv2d(out_features, out_features, 1, bias=False),
    )


def upconv_separable_block(in_features, out_features, bias=True, kernel=3, stride=2, padding=1, output_padding=1, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_features, in_features, 1),
        nn.ConvTranspose2d(
            in_features,
            out_features,
            kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=gcd(in_features, out_features),
            bias=bias,
            **kwargs
        ),
    )


class Activate(nn.Module):
    def forward(self, input: torch.Tensor):
        # return input.clamp(0.0, 1.0)
        return input.tanh().add(1).true_divide(2)


class StyleTransferFastOpt(nn.Module):
    def __init__(self, n_residual=15):
        super().__init__()
        self.n_residual = n_residual
        residual_stack = [ResidualSeparableBlock(64) for _ in range(self.n_residual)]
        self.layers = nn.Sequential(
            conv_separable_block(3, 24),
            nn.InstanceNorm2d(24, affine=True),
            nn.ReLU(),
            conv_separable_block(24, 64),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            *residual_stack,
            upconv_separable_block(64, 24),
            nn.ReLU(),
            upconv_separable_block(24, 3),
            Activate()
        )

    def forward(self, x):
        return self.layers(x)


model = StyleTransferFastOpt()
saved_state = torch.load('saves/StyleTransferFastOpt/e020.pth', map_location='cpu')
model.load_state_dict(saved_state['model_state_dict'])
model.to('cuda')
model.eval()


av.logging.set_level(av.logging.INFO)
with av.open('dragonguard.s1.mkv', mode="w") as container:
    stream = container.add_stream('ffv1', rate=30)
    stream.width = 1920
    stream.height = 1080
    stream.pix_fmt = "yuv420p"
    stream.options = {}
    with torch.set_grad_enabled(False):
        for pkt in tqdm(sampler, total=sampler.size):
            frames = pkt[0]['frames'].squeeze(0).permute((0, 3, 1, 2))
            frames = model\
                .forward(frames)\
                .mul(255)\
                .to(device='cpu', dtype=torch.uint8)\
                .permute((0, 2, 3, 1))
            for frame_idx in range(frames.size()[0]):
                frame = av.VideoFrame.from_ndarray(frames[frame_idx].numpy(), format="rgb24")
                frame.pict_type = "NONE"
                for packet in stream.encode(frame):
                    container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

# ffmpeg -i dragonguard.mkv -i dragonguard.s1.mkv -map 0:a -c:a libopus -b:a 96k\
# -map 1:v -c:v libvpx-vp9 -crf 30 -b:v 0 dragonguard.s1.webm
# ffmpeg -i dragonguard.mkv -i dragonguard.s1.mkv -map 0:a -c:a copy -map 1:v -c:v libx264 -crf 20 dragonguard.s1.mp4
