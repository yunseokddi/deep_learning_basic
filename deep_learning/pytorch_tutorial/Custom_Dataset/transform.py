#Rescle: 이미지 크기 조절
#RandomCrop: 이미지 무작위 자르기
#ToTesor: numpy image -> torch image

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Rescale(object):
    '''샘플 데이터의 크기를 조절

    output_size
    -> tuple이면 해당 tuple이 output_size가 되고,
       int라면 비율을 유지하면서 길이가 작은 쪽이 output_size가 된다.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h,w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size

            else:
                new_h, new_w = self.output_size, self.output_size * w/h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w/w, new_h/h]

        return {'image':img, 'landmarks':landmarks}

class RandomCrop(object):
    '''
        sample data를 무작위로 자른다.

        output_size:
                    int라면 정사각형으로 output이 나온다.

    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            assert len(output_size) == 2
            self.output_size = (output_size, output_size)

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h,w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top:top+new_h, left: left+new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks' :landmarks}


class ToTensor(object):

    '''
        numpy -> tensor
    '''


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}





