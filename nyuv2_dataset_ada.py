import torch.utils.data as data
import numpy as np
from PIL import Image
from path import Path
from constants import *
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip, CenterCrop, ColorJitter
import torch, time, os
import random
import scipy.ndimage as ndimage
import scipy.io as sio

num = 795 #22309#795
beta = 0.5  # for wls_res shift to avoid 0

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w, :]

        landmarks = landmarks[top: top + new_h,
                    left: left + new_w]

        return image, landmarks


class CropCenter(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        inputs = inputs[y1: y1 + th, x1: x1 + tw]
        target = target[y1: y1 + th, x1: x1 + tw]
        return inputs, target


class RandomCropRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    A crop is done to keep same image ratio, and no black pixels
    angle: max angle of the rotation, cannot be more than 180 degrees
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, angle, diff_angle=0, order=2):
        self.angle = angle
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, sample):
        inputs, target = sample
        h, w, _ = inputs.shape

        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2

        angle1_rad = angle1 * np.pi / 180

        inputs = ndimage.interpolation.rotate(inputs, angle1, reshape=True, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=True, order=self.order)

        # keep angle1 and angle2 within [0,pi/2] with a reflection at pi/2: -1rad is 1rad, 2rad is pi - 2 rad
        angle1_rad = np.pi / 2 - np.abs(angle1_rad % np.pi - np.pi / 2)

        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c_diag = h / np.sqrt(h * h + w * w)
        s_diag = w / np.sqrt(h * h + w * w)

        ratio = 1. / (c1 + w / float(h) * s1)

        crop = CropCenter((int(h * ratio), int(w * ratio)))
        return crop(inputs, target)


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, landmarks1, landmarks2 = sample[0], sample[1], sample[2]
        if np.random.randn() > 0.5:
            image = image[:, ::-1, :]
            landmarks1 = landmarks1[:, ::-1]
            landmarks2 = landmarks2[:, ::-1]

        return image, landmarks1, landmarks2


class NYUv2Dataset(data.Dataset):
    def __init__(self, root='/datasets/nyud_v2/', seed=None, train=True):

        np.random.seed(seed)
        self.root = Path(root)
        self.train = train
        if train:
            self.rgb_paths = [root + 'train/images/' + d for d in os.listdir(root + 'train/images/')]  # for labeled dataset
            # Randomly choose num images without replacement
            self.rgb_paths = np.random.choice(self.rgb_paths, num, False)  # for labeled dataset
        else:
            # for test
            self.rgb_paths = [root + 'test/images/' + d for d in os.listdir(root + 'test/images/')]
    

        self.augmentation = Compose([RandomHorizontalFlip()])
        # for labeled dataset
        self.rgb_transform_H = Compose([ToPILImage(), Resize((480, 640)), ToTensor()])
        self.depth_transform_H = Compose([ToPILImage(), Resize((120, 160)), ToTensor()])
        self.rgb_transform_L = Compose([ToPILImage(), Resize((360, 480)), ToTensor()])
        self.depth_transform_L = Compose([ToPILImage(), Resize((90, 120)), ToTensor()])

        if self.train:
            self.length = num  # len(self.rgb_paths)
        else:
            self.length = len(self.rgb_paths)  # for test

    def __getitem__(self, index):
        path = self.rgb_paths[index]
        rgb = Image.open(path)
        if self.train == True:
            dep = sio.loadmat(
                path.replace('images_withtest', 'wls/depths_withtest').replace('jpg','mat'))  # for WLS data images_projected_filled
            dep_base = sio.loadmat(
                path.replace('images_withtest', 'wls/depths_base_withtest').replace('jpg', 'mat'))  # for WLS data

        else:
            dep = sio.loadmat(path.replace('images', 'depths').replace('jpg', 'mat'))

        if self.train == True:
            depth = dep['I']  # # for WLS data
            depth_base = dep_base['base']  # for WLS data

            depth = depth.astype(np.float32)  # normalized
            depth_base = depth_base.astype(np.float32)  # depth = depth.astype(np.float32) / 10  #  normalized

        else:
            depth = dep['temp']# for labeled dataset
            depth = depth.astype(np.float32) / 10.  # normalized

        if self.train:
          
            rgb, depth, depth_base = np.array(rgb), np.array(depth)[:, :, None], np.array(depth_base)[:,:,None]
            rgb, depth, depth_base= self.augmentation((rgb, depth, depth_base))
            a = self.rgb_transform_L(rgb)
            b = self.depth_transform_L(depth_base).squeeze(-1)
            c = self.rgb_transform_H(rgb)
            d = self.depth_transform_H(depth).squeeze(-1)
        
            return a, b, c, d
        else:
            return Compose([Resize((360, 480)), ToTensor()])(rgb), Compose([Resize((480, 640)), ToTensor()])(rgb), ToTensor()(depth)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    # Testing
    dataset = NYUv2Dataset()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
