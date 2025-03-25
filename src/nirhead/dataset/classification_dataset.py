import pathlib, PIL.Image, zipfile, os, json, cv2
import torchvision
import numpy as np
from eg3d.training.dataset import pyspng
from elias.util.io import resize_img


class ClassificationDataSet:
    def __init__(self, root, resolution=None, mode=None, labelclasses=None, subdir="", flip=False, inference=False):
        self.root = root
        if isinstance(self.root, str):
            self.root = pathlib.Path(self.root)
        self.resolution = resolution
        self.mode = mode.strip().lower().replace("l", "gray").replace("bgr", "rgb")
        self.labelclasses = labelclasses
        self.subdir = subdir.strip()
        self.flip = flip
        self.inference = inference

        self._zipfile = None
        self._type = None
        if os.path.isfile(self.root) and self.root.name.lower().endswith(".zip"):
            self._type = "zip"
        elif os.path.isdir(self.root):
            self._type = "dir"
        else:
            return

        if self._type == "zip":
            self.images = [f for f in self._get_zipfile().namelist() if f.replace("\\", "/").startswith((self.subdir+"/" if self.subdir else "")) and f.endswith(".png")]
        else:
            self.images = [str(f.absolute())[len(str(self.root.absolute()))+1:] for f in (self.root / self.subdir).rglob("*.png")]

        labeldata = {}
        if not self.inference:
            with self._open_file("labels.json") as f:
                labeldata = json.load(f)
            self.labels = []
            for file in self.images:
                file = file.replace("\\", "/")
                assert(file in labeldata.keys())
                l = [float(labeldata[file][cl]) for cl in self.labelclasses if cl in labeldata[file].keys()]
                assert(len(l) == len(self.labelclasses))
                self.labels.append(np.array(l, dtype=np.float32))

        assert(self.inference or len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images) if not self.flip else len(self.images)*2

    def __getitem__(self, idx):
        flip = False
        if self.flip:
            flip = idx % 2 == 1
            idx = idx // 2
        img = self._load_image(self.images[idx], flip=flip)
        
        if self.inference:
            return img

        label = self.labels[idx]
        return img, label

    def get_image_path(self, idx):
        if self.flip:
            idx = idx // 2
        return self.images[idx]

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.root)
        return self._zipfile

    def _open_file(self, file):
        if self._type == 'dir':
            return open(self.root / file, 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(file, 'r')
        return None

    def _load_image(self, file, flip=False):
        with self._open_file(file) as f:
            if pyspng is not None and file.lower()[-4:] == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        if self.resolution is not None:
            if image.shape[2] == 1:
                image = resize_img(image[..., 0], self.resolution / image.shape[0])[..., None]
            else:
                image = resize_img(image, self.resolution / image.shape[0])
        if not self.mode is None:
            if image.shape[2] > 1 and self.mode == "gray":
                image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
            elif image.shape[2] == 1 and self.mode == "rgb":
                image = np.stack((image[:,:,0],)*3, axis=-1)

        image = image.transpose(2, 0, 1)  # HWC => CHW

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint32:
            image = image.astype(np.float32) / 4294967295.0
        elif image.dtype == np.float16 or image.dtype == np.float64:
            image = image.astype(np.float32)

        if flip:
            image = image[:,:,::-1]

        image = image * 2.0 - 1.0 # normalize
        return image

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
