import glob

from tqdm import tqdm
from PIL import Image


imgs = sorted(glob.glob('./*.jpg'))

def norm_size(img, size=128, bgc=(0,0,0)):
    w, h = img.size
    max_size = max(w, h)
    x0 = (max_size - w) // 2
    y0 = (max_size - h) // 2
    x1 = x0 + w
    y1 = y0 + h
    canvas = Image.new('RGB', (max_size,max_size), bgc)
    canvas.paste(img, (x0,y0,x1,y1))

    canvas = canvas.resize((size*2,size*2), resample=Image.BOX)
    canvas = canvas.resize((size,size), resample=Image.BICUBIC)

    return canvas

for i, path in enumerate(tqdm(imgs)):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    sketch = norm_size(img.crop((0,0,w//2,h//2)), bgc=(255,255,255))
    block = norm_size(img.crop((w//2,0,w,h//2)))
    anime = norm_size(img.crop((0,h//2,w//2,h)))
    ray = norm_size(img.crop((w//2,h//2,w,h)))
    sketch.convert('L').save(f'anime_sketch/{i+1}.png')
    block.save(f'anime_block/{i+1}.png')
    anime.save(f'anime/{i+1}.png')
    ray.save(f'anime_ray/{i+1}.png')
