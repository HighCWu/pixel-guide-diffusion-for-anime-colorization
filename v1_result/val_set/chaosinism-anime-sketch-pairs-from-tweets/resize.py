import glob

from tqdm import tqdm
from PIL import Image, ImageDraw


imgs = sorted(glob.glob('./src/anime/*.jpg'))

def norm_size(img, size=128, bgc=(0,0,0)):
    w, h = img.size
    max_size = max(w, h)
    x0 = (max_size - w) // 2
    y0 = (max_size - h) // 2
    x1 = x0 + w
    y1 = y0 + h
    canvas = Image.new('RGB', (max_size,max_size), bgc)
    canvas.paste(img, (x0,y0,x1,y1))

    draw = ImageDraw.Draw(canvas) 
    draw.line((x0-2,0,x0-1,max_size), fill=0)
    draw.line((0,y0-2,max_size,y0-1), fill=0)
    draw.line((x1+1,0,x1+2,max_size), fill=0)
    draw.line((0,y1+1,max_size,y1+2), fill=0)

    canvas = canvas.resize((size*2,size*2), resample=Image.BOX)
    canvas = canvas.resize((size,size), resample=Image.BICUBIC)

    return canvas

for i, path in enumerate(tqdm(imgs)):
    path = path.replace('\\', '/')
    anime = Image.open(path).convert('RGB')
    anime = norm_size(anime)
    sketch = Image.open(path.replace('/anime/', '/anime_sketch/')).convert('RGB')
    sketch = norm_size(sketch, bgc=(255,255,255))
    sketch.convert('L').save(path.replace('/anime/', '/anime_sketch/').replace('/src/', '/resized/').replace('.jpg', '.png'))
    anime.save(path.replace('/src/', '/resized/').replace('.jpg', '.png'))
