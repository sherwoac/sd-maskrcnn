import os
import sys
import logging
from tqdm import tqdm

import cv2
import skimage.io
import numpy as np
import tensorflow as tf

from mrcnn import model as modellib, visualize, utils


"""
TargetDataset creates a Matterport dataset for a directory of
target images, used for target detection branch training.
Directory structure must be as follows:
$base_path/
    target.json
    images/ (target images here)
        ...
    piles/ (color pile images here)
        ...
    masks/ (masks corresponding to piles)

target.json must contain a list of tuples with the following format:
(target_path, pile_path, target_index)

In this class, image_id does not map one-to-one to different piles, rather
to different pile/target pairs.
"""

class TargetDataset(utils.Dataset):
    def __init__(self, base_path, images='piles', masks='masks', targets='images'):
        # omg fix this above before u get confused!!!
        assert base_path != "", "You must provide the path to a dataset!"
        self.targets = targets
        self.images = images
        self.masks = masks
        self.base_path = base_path
        self.data_tuples = None
        super().__init__()

    def load(self, imset=None):
        self.add_class('clutter', 1, 'fg')
        import json
        self.data_tuples = json.load(open(os.path.join(self.base_path, 'target.json')))

        # Provide optional index file. NOTE: This operates on the JSON files!
        if imset:
            self.image_id = np.load(imset)
        else:
            self.image_id = list(range(len(self.data_tuples)))

        for i in self.image_id:
            pile_path = os.path.join(self.base_path, self.images,
                                     self.data_tuples[i][1])
            target_path = os.path.join(self.base_path, self.images,
                                       self.data_tuples[i][1])
            self.add_image('clutter', image_id=i, path=pile_path,
                           target_path=target_path)


    def load_image(self, image_id):
        image = skimage.io.imread(os.path.join(self.base_path, self.images,
                                               self.image_id[image_id][1]))
        return image


    def load_target(self, image_id):
        """Returns target image"""
        target = skimage.io.imread(os.path.join(self.base_path, self.targets,
                                                self.image_id[image_id][0]))

        return target

    def load_target_index(self, image_id):
        """Returns index of target mask"""
        target_index = int(self.image_id[image_id][2]) - 1 # because we no longer consider the background
        return target_index

    def load_mask(self, image_id):
        """Returns masks corresponding to pile, class IDs"""
        Is = []
        all_masks = skimage.io.imread(os.path.join(self.base_path, self.masks,
                                                   self.image_id[image_id][1]))
        for i in np.arange(1,np.max(all_masks)+1):
            I = all_masks == i # We ignore the background, so the first instance is 0-indexed.
            if np.any(I):
                I = I[:,:,np.newaxis]
                Is.append(I)
        if len(Is) > 0:
            mask = np.concatenate(Is, 2)

        class_ids = np.array([1 for _ in range(mask.shape[2])])
        return mask, class_ids.astype(np.int32)

    @property
    def indices(self):
        return self._image_ids



"""
ImageDataset creates a Matterport dataset for a directory of
images in order to ensure compatibility with benchmarking tools
and image resizing for networks.
Directory structure must be as follows:
$base_path/
    depth_ims/ (Depth images here)
        image_000000.png
        image_000001.png
        ...
    color_ims/ (Color images here)
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/ (GT segmasks here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class ImageDataset(utils.Dataset):
    def __init__(self, base_path, images, masks):
        assert base_path != "", "You must provide the path to a dataset!"

        self.base_path = base_path
        self.images = images
        self.masks = masks
        super().__init__()

    def load(self, imset, augment=False):

        # Load the indices for imset.
        split_file = os.path.join(self.base_path, '{:s}'.format(imset))
        self.image_id = np.load(split_file)
        self.add_class('clutter', 1, 'fg')

        flips = [1, 2, 3]
        for i in self.image_id:
            if 'numpy' in self.images:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.npy'.format(i))
            else:
                p = os.path.join(self.base_path, self.images,
                                'image_{:06d}.png'.format(i))
            self.add_image('clutter', image_id=i, path=p, width=512, height=384)

            if augment:
                for flip in flips:
                    self.add_image('clutter', image_id=i, path=p, width=512, height=384, flip=flip)

    def flip(self, image, flip):
        # flips during training for augmentation

        if flip == 1:
            image = image[::-1,:,:]
        elif flip == 2:
            image = image[:,::-1,:]
        elif flip == 3:
            image = image[::-1,::-1,:]
        return image

    def load_image(self, image_id):
        # loads image from path
        if 'numpy' in self.images:
            image = np.load(self.image_info[image_id]['path']).squeeze()
        else:
            image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "clutter":
            return info["path"] + "-{:d}".format(info["flip"])
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        # loads mask from path
        info = self.image_info[image_id]
        _image_id = info['id']
        Is = []
        file_name = os.path.join(self.base_path, self.masks,
          'image_{:06d}.png'.format(_image_id))

        all_masks = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        for i in np.arange(1,np.max(all_masks)+1):
            I = all_masks == i # We ignore the background, so the first instance is 0-indexed.
            if np.any(I):
                I = I[:,:,np.newaxis]
                Is.append(I)
        if len(Is) > 0:
            mask = np.concatenate(Is, 2)
        else:
            mask = np.zeros([info['height'], info['width'], 0], dtype=np.bool)

        class_ids = np.array([1 for _ in range(mask.shape[2])])
        return mask, class_ids.astype(np.int32)

    @property
    def indices(self):
        return self.image_id
