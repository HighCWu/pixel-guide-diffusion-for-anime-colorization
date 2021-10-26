from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, guide_size=0, guide_dir=None, crop_size=0, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param guide_size: the size to which images are resized for guide tensors.
    :param guide_dir: a dataset directory for guide tensors.
    :param crop_size: the size to which images are resized and cropped.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    guide_files = None
    if guide_dir:
        guide_files = _list_image_files_recursively(guide_dir)
    guide_files2 = _list_image_files_recursively('data/danbooru2017/anime_sketch_noise')
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        guide_resolution=guide_size,
        guide_paths=guide_files,
        guide_paths2=guide_files2,
        crop_resolution=crop_size,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return sorted(results)


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, guide_resolution=0, guide_paths=None, guide_paths2=None, crop_resolution=0, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.guide_resolution = guide_resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_guides = guide_paths[shard:][::num_shards] if guide_paths else None
        self.local_guides2 = guide_paths2[shard:][::num_shards] if guide_paths else None
        self.crop_resolution = crop_resolution if crop_resolution > 0 else resolution
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images) * 1000000

    def __getitem__(self, idx):
        idx = idx % len(self.local_images)
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.crop_resolution) // 2
        crop_x = (arr.shape[1] - self.crop_resolution) // 2
        arr = arr[crop_y : crop_y + self.crop_resolution, crop_x : crop_x + self.crop_resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}

        if self.local_guides:
            path = self.local_guides[idx] if np.random.rand() < 0.5 else self.local_guides2[idx]
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()

            # We are not on a new enough PIL to support the `reducing_gap`
            # argument, which uses BOX downsampling at powers of two first.
            # Thus, we do it by hand to improve downsample quality.
            while min(*pil_image.size) >= 2 * self.guide_resolution:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = self.guide_resolution / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            crop_resolution = self.guide_resolution // self.resolution * self.crop_resolution

            guide_arr = np.array(pil_image.convert("L"))[...,None] # np.array(pil_image.convert("RGB"))
            
            # extra noise
            if np.random.rand() < 0.5:
                w, h = guide_arr.shape[:2][::-1]
                a = np.random.randint(2,12)
                mean = np.asarray(
                    Image.fromarray(
                        np.random.randint(0,255,[a,a],dtype='uint8')
                    ).resize([w,h], Image.NEAREST)
                ).astype('float32') / 255.0 * 2 - 1
                std = np.asarray(
                    Image.fromarray(
                        np.random.randint(0,255,[a,a],dtype='uint8')
                    ).resize([w, h], Image.NEAREST)
                ).astype('float32') / 255.0 * 7.5 + 0.125
                guide_arr = (guide_arr - mean[...,None]) * std[...,None]
                
            crop_y = (guide_arr.shape[0] - crop_resolution) // 2
            crop_x = (guide_arr.shape[1] - crop_resolution) // 2
            guide_arr = guide_arr[crop_y : crop_y + crop_resolution, crop_x : crop_x + crop_resolution]
            guide_arr = guide_arr.astype(np.float32) / 127.5 - 1
            
            out_dict["guide"] = np.transpose(guide_arr, [2, 0, 1]).astype('float32')

        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict
