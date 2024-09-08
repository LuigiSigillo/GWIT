# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import config

import legacy
from training.dataset import Image2EEG2ImageDataset

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(test_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // test_set.image_shape[2], 8, 32) # For each class 32 samples
    gh = np.clip(5120 // test_set.image_shape[1], 5, 40) # Number of classes 40

    # No labels => show random subset of training samples.
    if not test_set.has_labels:
        all_indices = list(range(len(test_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(test_set)):
            # label = tuple(test_set.get_details(idx).raw_label.flat[::-1])
            label = tuple([test_set.labels[idx].detach().cpu().item()])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)
        
        # print(label_groups)
        # print(len(label_groups.keys()))

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[test_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    data: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # os.makedirs(outdir+'/{}'.format(class_idx), exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    test_set = Image2EEG2ImageDataset(path=data, resolution=config.image_height)
    # test_set_iterator = torch.utils.data.DataLoader(dataset=test_set, sampler=test_set_sampler, batch_size=args.batch_size)
    batch_gpu = 4

    for seed_idx, seed in enumerate(seeds):
        print('Reconstructing image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        grid_size, images, labels = setup_snapshot_image_grid(test_set=test_set, random_seed=seed)
        save_image_grid(images, f'{outdir}/seed{seed:04d}_real.pdf', drange=[0,255], grid_size=grid_size)
        # grid_size, images, labels = setup_snapshot_image_grid(test_set=test_set)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        # import pdb; pdb.set_trace()
        images = torch.cat([G(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        # images = torch.cat([G(z=z, c=c, noise_mode='random').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, f'{outdir}/seed{seed:04d}_reconst.pdf', drange=[-1,1], grid_size=grid_size)

    # Synthesize the result of a W projection.
    # if projected_w is not None:
    #     if seeds is not None:
    #         print ('warn: --seeds is ignored when using --projected-w')
    #     print(f'Generating images from projected W "{projected_w}"')
    #     ws = np.load(projected_w)['w']
    #     ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
    #     assert ws.shape[1:] == (G.num_ws, G.w_dim)
    #     for idx, w in enumerate(ws):
    #         img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
    #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #         img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
    #     return

    # if seeds is None:
    #     ctx.fail('--seeds option is required when not using --projected-w')

    # # Labels.
    # label = torch.zeros([1, G.c_dim], device=device)
    # if G.c_dim != 0:
    #     if class_idx is None:
    #         ctx.fail('Must specify class label with --class when using a conditional network')
    #     label[:, class_idx] = 1
    # else:
    #     if class_idx is not None:
    #         print ('warn: --class=lbl ignored when running on an unconditional network')

    # # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    #     z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    #     img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #     PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{class_idx}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
