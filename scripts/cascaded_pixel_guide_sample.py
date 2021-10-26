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
    pg_model_and_diffusion_defaults,
    pg_create_model_and_diffusion,
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
    model, diffusion = pg_create_model_and_diffusion(
        **args_to_dict(args, pg_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating model2...")
    args.num_channels = args.num_channels2
    args.use_attention = args.use_attention2
    model2, diffusion2 = pgsr_create_model_and_diffusion(
        **args_to_dict(args, pgsr_model_and_diffusion_defaults().keys())
    )
    model2.load_state_dict(
        dist_util.load_state_dict(args.model_path2, map_location="cpu")
    )
    model2.to(dist_util.dev())
    model2.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
        guide_dir=args.guide_dir,
        guide_size=args.guide_size,
        deterministic=True,
    )

    if args.seed > -1:
        th.manual_seed(args.seed)

    logger.log("creating samples...")
    os.makedirs('sample', exist_ok=True)
    i = 0
    while i * args.batch_size < args.num_samples:
        if dist.get_rank() == 0:
            target, model_kwargs = next(data)
            target = target.to(dist_util.dev())
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

            with th.no_grad():
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                model_kwargs["low_res"] = sample
                sample_fn2 = (
                    diffusion2.p_sample_loop if not args.use_ddim else diffusion2.ddim_sample_loop
                )
                sample2 = sample_fn2(
                    model2,
                    (args.batch_size, 3, args.large_size, args.large_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                guide = model_kwargs["guide"]
                h, w = guide.shape[2:]
                guide = guide.clamp(-1,1).repeat(1,3,1,1)
                sample = th.nn.functional.interpolate(sample.clamp(-1,1), size=(h, w))
                sample2 = th.nn.functional.interpolate(sample2.clamp(-1,1), size=(h, w))
                target = th.nn.functional.interpolate(target.clamp(-1,1), size=(h, w))

                # images = th.cat([guide, sample, sample2, target], 0)
                images = th.cat([guide, sample2, target], 0)
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


def create_argparser():
    defaults = dict(
        data_dir="",
        guide_dir="",
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        use_ddim=False,
        base_samples="",
        model_path="",
        seed=-1,
    )
    defaults.update(pg_model_and_diffusion_defaults())
    defaults.update(pgsr_model_and_diffusion_defaults())
    defaults.update(dict(
        num_channels2=128,
        use_attention2=True,
        model_path2="",
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
