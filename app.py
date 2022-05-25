"""
A Gradio Blocks Demo App.
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import gradio as gr
import argparse
import os
import glob

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from PIL import Image, ImageDraw
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

MODEL_FLAGS="--image_size=32 --small_size=32 --large_size=128 --guide_size=128 --num_channels=128 --num_channels2=64 --num_res_blocks=3 --learn_sigma=True --dropout=0.0 --use_attention2=False"
DIFFUSION_FLAGS="--diffusion_steps=4000 --noise_schedule=cosine"
TEST_FLAGS="--batch_size=1 --seed=233 --num_samples=4"
OTHER_FLAGS = '''\
--timestep_respacing=16 \
--use_ddim=False \
--model_path=./danbooru2017_guided_log/ema_0.9999_360000.pt \
--model_path2=./danbooru2017_guided_sr_log/ema_0.9999_360000.pt'''
OTHER_FLAGS = OTHER_FLAGS.replace('\r\n', ' ').replace('\n', ' ')
flags = OTHER_FLAGS.split(' ') + MODEL_FLAGS.split(' ') + DIFFUSION_FLAGS.split(' ') + TEST_FLAGS.split(' ')


def norm_size(img, size=128):
    img = img.convert('L')
    w, h = img.size
    max_size = max(w, h)
    x0 = (max_size - w) // 2
    y0 = (max_size - h) // 2
    x1 = x0 + w
    y1 = y0 + h
    canvas = Image.new('L', (max_size,max_size), 255)
    canvas.paste(img, (x0,y0,x1,y1))

    draw = ImageDraw.Draw(canvas) 
    draw.line((x0-2,0,x0-1,max_size), fill=0)
    draw.line((0,y0-2,max_size,y0-1), fill=0)
    draw.line((x1+1,0,x1+2,max_size), fill=0)
    draw.line((0,y1+1,max_size,y1+2), fill=0)

    canvas = canvas.resize((size*2,size*2), resample=Image.BOX)
    canvas = canvas.resize((size,size), resample=Image.BICUBIC)

    return canvas


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


@th.inference_mode()
def main():
    args = create_argparser().parse_args(flags)

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

    def inference(img, seed):
        th.manual_seed(int(seed))
        sketch = norm_size(img, size=128)
        sketch = np.asarray(sketch).astype(np.float32) / 127.5 - 1
        sketch = th.from_numpy(sketch).float()[None,None].to(dist_util.dev())
        model_kwargs = { "guide": sketch }
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
        out = (sample2[0].clamp(-1,1).cpu().numpy() + 1) / 2 * 255
        out = np.uint8(out)
        out = out.transpose([1,2,0])
        out = Image.fromarray(out)

        return out

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        seed_in = gr.Number(
                            value=233, 
                            label='Seed'
                        )
                    with gr.Row():
                        sketch_in = gr.Image(
                            type="pil", 
                            label="Sketch"
                        )
                    with gr.Row():
                        generate_button = gr.Button('Generate')
                    with gr.Row():
                        example_sketch_paths = [[p] for path in sorted(glob.glob('docs/imgs/anime_sketch/*.png'))]
                        example_sketch = gr.Dataset(
                            components=[sketch_in], 
                            samples=[example_sketch_paths]
                        )
                    with gr.Row():
                        example_real_paths = [[p] for path in sorted(glob.glob('docs/imgs/anime/*.png'))]
                        example_real = gr.Dataset(
                            components=[sketch_in], 
                            samples=[example_real_paths]
                        )
        
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        colorized_out = gr.Image(
                            type="pil", 
                            label="Colorization result"
                        )
        generate_button.click(
            inference, inputs=[sketch_in, seed_in], outputs=[colorized_out]
        )
        example_sketch.click(
            fn=lambda examples: gr.Image.update(value=examples[0]), 
            inputs=example_sketch, 
            outputs=example_sketch.components
        )
        
        demo.launch()


if __name__ == '__main__':
    main()
