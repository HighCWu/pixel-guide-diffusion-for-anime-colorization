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


def norm_size(img, size=128, add_edges=True):
    img = img.convert('L')
    w, h = img.size
    if w != h:
        scale = 1024 / max(img.size)
        img = img.resize([int(round(s*scale)) for s in img.size])
        w, h = img.size
        max_size = max(w, h)
        x0 = (max_size - w) // 2
        y0 = (max_size - h) // 2
        x1 = x0 + w
        y1 = y0 + h
        canvas = Image.new('L', (max_size,max_size), 255)
        canvas.paste(img, (x0,y0,x1,y1))

        if add_edges:
            draw = ImageDraw.Draw(canvas) 
            draw.line((x0-5,0,x0-1,max_size), fill=0)
            draw.line((0,y0-5,max_size,y0-1), fill=0)
            draw.line((x1+1,0,x1+5,max_size), fill=0)
            draw.line((0,y1+1,max_size,y1+5), fill=0)

        img = canvas
    img = img.resize((size,size), resample=Image.LANCZOS)

    return img


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

    def inference(img, seed, add_edges):
        th.manual_seed(int(seed))
        sketch = sketch_out = norm_size(img, size=128, add_edges=add_edges)
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

        return sketch_out, out

    with gr.Blocks() as demo:
        gr.Markdown('''<center><h1>Anime-Colorization</h1></center>
<h2>Colorize your anime sketches with this app.</h2>
This is a Gradio Blocks app of 
<a href="https://github.com/HighCWu/pixel-guide-diffusion-for-anime-colorization">
HighCWu/pixel-guide-diffusion-for-anime-colorization
</a>.<br />
(PS: Training Datasets are made from <a href="https://www.kaggle.com/datasets/wuhecong/danbooru-sketch-pair-128x">
HighCWu/danbooru-sketch-pair-128x
</a> which processed real anime images to sketches by 
<a href="https://github.com/lllyasviel/sketchKeras">SketchKeras</a>.
So the model is not very sensitive to some different styles of sketches,
and the colorized results of such sketches are not very good.)
''')
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        seed_in = gr.Number(
                            value=233, 
                            label='Seed'
                        )
                    with gr.Row():
                        edges_in = gr.Checkbox(
                            label="Add Edges"
                        )
                    with gr.Row():
                        sketch_in = gr.Image(
                            type="pil", 
                            label="Sketch"
                        )
                    with gr.Row():
                        generate_button = gr.Button('Generate')
                    with gr.Row():
                        gr.Markdown('Click to add example as input.👇')
                    with gr.Row():
                        example_sketch_paths = [[p] for p in sorted(glob.glob('docs/imgs/anime_sketch/*.png'))]
                        example_sketch = gr.Dataset(
                            components=[sketch_in], 
                            samples=example_sketch_paths
                        )
                    with gr.Row():
                        gr.Markdown('These are expect real outputs.👇')
                    with gr.Row():
                        example_real_paths = [[p] for p in sorted(glob.glob('docs/imgs/anime/*.png'))]
                        example_real = gr.Dataset(
                            components=[sketch_in], 
                            samples=example_real_paths
                        )
        
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            sketch_out = gr.Image(
                                type="pil", 
                                label="Input"
                            )
                        with gr.Column():
                            colorized_out = gr.Image(
                                type="pil", 
                                label="Colorization Result"
                            )
                    with gr.Row():
                        gr.Markdown(
                            'Here are some samples 👇 [top: sketch, center: generated, bottom: real]'
                        )
                    with gr.Row():
                        gr.Image(
                            value="docs/imgs/sample.png",
                            type="filepath", 
                            interactive=False,
                            label="Samples"
                        )
        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.anime-colorization" alt="visitor badge"/></center>'
        )

        generate_button.click(
            inference, inputs=[sketch_in, seed_in, edges_in], outputs=[sketch_out, colorized_out]
        )
        example_sketch.click(
            fn=lambda examples: gr.Image.update(value=examples[0]), 
            inputs=example_sketch, 
            outputs=example_sketch.components
        )
        
        demo.launch()

if __name__ == '__main__':
    main()
