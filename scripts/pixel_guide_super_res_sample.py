"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from torchvision import utils
from pixel_guide_diffusion import dist_util, logger
from pixel_guide_diffusion.image_datasets import load_data
from pixel_guide_diffusion.script_util import (
    pgsr_model_and_diffusion_defaults,
    pgsr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = pgsr_create_model_and_diffusion(
        **args_to_dict(args, pgsr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        guide_dir=args.guide_dir,
        guide_size=args.guide_size,
        crop_size=args.crop_size,
        deterministic=True,
    )

    logger.log("creating samples...")
    os.makedirs('sample', exist_ok=True)
    i = 0
    while i * args.batch_size < args.num_samples:
        if dist.get_rank() == 0:
            target, model_kwargs = next(data)
            target = target.to(dist_util.dev())
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            model_kwargs["low_res"] = th.nn.functional.interpolate(target, args.small_size, mode="area").detach()

            with th.no_grad():
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.crop_size, args.crop_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                guide = model_kwargs["guide"]
                low_res = model_kwargs["low_res"]
                h, w = guide.shape[2:]
                guide = guide.clamp(-1,1).repeat(1,3,1,1)
                low_res = th.nn.functional.interpolate(low_res.clamp(-1,1), size=(h, w))
                sample = th.nn.functional.interpolate(sample.clamp(-1,1), size=(h, w))
                target = th.nn.functional.interpolate(target.clamp(-1,1), size=(h, w))

                images = th.cat([guide, low_res, sample, target], 0)
                utils.save_image(
                    images,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=args.batch_size,
                    normalize=True,
                    range=(-1, 1),
                )

                i += 1
                logger.log(f"created {i * args.batch_size} samples")

    logger.log("sampling complete")


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False, guide_dir='', guide_size=0, crop_size=0, deterministic=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        guide_dir=guide_dir,
        guide_size=guide_size,
        crop_size=crop_size,
        deterministic=deterministic,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["low_res"] = th.nn.functional.interpolate(large_batch, scale_factor=small_size/large_size, mode="area").detach()
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        guide_dir="",
        crop_size=128,
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(pgsr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
