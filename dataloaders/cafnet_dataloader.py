import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

from dataloaders.distributed_sampler_no_evenly_divisible import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

class CaFNetDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.image_path = args.train_image_path
            self.radar_path = args.train_radar_path
            self.ground_truth_path = args.train_ground_truth_path
            self.box_pos_path = args.train_box_pos_path
            self.main_path = args.main_path

            image_paths = read_paths(self.image_path)
            radar_paths = read_paths(self.radar_path)
            ground_truth_paths = read_paths(self.ground_truth_path)
            box_pos_paths = read_paths(self.box_pos_path)

            self.training_samples = DataLoadPreprocess(args, mode, ground_truth_paths=ground_truth_paths, \
            image_paths=image_paths, radar_paths=radar_paths, box_pos_paths=box_pos_paths, main_path=self.main_path, \
            transform=preprocessing_transforms(mode))

            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)
        else:
            self.image_path = args.test_image_path
            self.radar_path = args.test_radar_path
            self.ground_truth_path = args.test_ground_truth_path
            self.main_path = args.main_path

            image_paths = read_paths(self.image_path)
            radar_paths = read_paths(self.radar_path)
            ground_truth_paths = read_paths(self.ground_truth_path)

            self.testing_samples = DataLoadPreprocess(args, mode, ground_truth_paths=ground_truth_paths, \
            image_paths=image_paths, radar_paths=radar_paths, box_pos_paths=None, main_path=self.main_path, \
            transform=preprocessing_transforms(mode))
            
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, ground_truth_paths=None, image_paths=None, radar_paths=None, box_pos_paths=None,\
                 main_path=None, transform=None, is_for_online_eval=False):
        self.args = args

        self.ground_truth_paths = ground_truth_paths
        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.box_pos_paths = box_pos_paths
        self.main_path = main_path

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
    
    def __getitem__(self, idx):

        focal = float(567.0)


        if self.mode == 'train':
            image_path = self.image_paths[idx]
            radar_path = self.main_path + self.radar_paths[idx]
            depth_path = self.main_path + self.ground_truth_paths[idx]
            box_pos_path = self.main_path + self.box_pos_paths[idx]
    
            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            width = image.shape[1]
            height = image.shape[0]

            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0

            radar_points = np.load(radar_path)
            box_pos_load = np.load(box_pos_path)

            box_pos = np.zeros((35, 4), dtype=np.int32)
            if len(box_pos_load) != 0:
                box_pos_load[box_pos_load < 0] = 0
                box_pos_load[:, 2][box_pos_load[:, 2] >= width] = width
                box_pos_load[:, 3][box_pos_load[:, 3] >= height] = height

                box_pos[:box_pos_load.shape[0]] = box_pos_load
            
            # map radar points to channels - depth, rcs, vx, vy, radar_box_align
            radar_channels = np.zeros((image.shape[0], image.shape[1], radar_points.shape[-1]-3), dtype=np.float32)
            radar_gt = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
            for i in range(radar_points.shape[0]):
                x = int(radar_points[i, 0])
                y = int(radar_points[i, 1])
                radar_depth = radar_points[i, 2]
                alignment = int(radar_points[i, -1])
                
                # generate radar channels
                if radar_channels[y, x, 0] == 0:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                elif radar_channels[y, x, 0] > radar_points[i, 2]:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                elif radar_channels[y, x, -1] == 0 and radar_points[i, -1] != 0:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                # generate radar alignment GT
                if alignment != 0:
                    x1, y1, x2, y2 = box_pos[alignment-1]
                else:
                    ext_h = self.args.patch_size[0]
                    ext_w = self.args.patch_size[1]
                    delta_x1 = np.minimum(x, ext_w)
                    delta_y1 = np.minimum(y, ext_h)
                    delta_x2 = np.minimum(image.shape[1]-x, ext_w)
                    delta_y2 = np.minimum(image.shape[0]-y, ext_h)
                    x1 = x - delta_x1
                    y1 = y - delta_y1
                    x2 = x + delta_x2
                    y2 = y + delta_y2

                distance_radar_ground_truth_depth = np.abs(depth_gt[y1:y2, x1:x2] - radar_depth * np.ones_like(depth_gt[y1:y2, x1:x2]))
                gt_label = np.where(
                    distance_radar_ground_truth_depth < self.args.max_dist_correspondence,
                    np.ones_like(depth_gt[y1:y2, x1:x2]),
                    np.zeros_like(depth_gt[y1:y2, x1:x2])
                )
                gt_label = np.where(
                    gt_label > 0,
                    gt_label,
                    np.zeros_like(gt_label)
                )
                radar_gt[y1:y2, x1:x2] = gt_label
            

            image, depth_gt, radar_channels, box_pos, radar_gt = self.random_crop(image, depth_gt, radar_channels, box_pos, radar_gt, \
                                                                        self.args.input_height, self.args.input_width)
            image, depth_gt, radar_channels, box_pos, radar_gt = self.train_preprocess(image, depth_gt, radar_channels, box_pos, radar_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'radar': radar_channels, 'box_pos': box_pos, 'radar_gt': radar_gt}
        
        else:

            image_path = self.image_paths[idx]
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
            # crop
            image = image[4:, ...] # (894, 1600, 3)

            depth_path =  self.main_path + self.ground_truth_paths[idx]
            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            # crop
            depth_gt = depth_gt[4:, ...]
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0

            radar_path = self.main_path + self.radar_paths[idx]
            radar_points = np.load(radar_path)
            radar_channels = np.zeros((image.shape[0], image.shape[1], radar_points.shape[-1]-3), dtype=np.float32)
            for i in range(radar_points.shape[0]):
                x = int(radar_points[i, 0])
                y = int(radar_points[i, 1])
                if radar_channels[y, x, 0] == 0:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                elif radar_channels[y, x, 0] > radar_points[i, 2]:
                    radar_channels[y, x] = radar_points[i, 2:-1]
                elif radar_channels[y, x, -1] == 0 and radar_points[i, -1] != 0:
                    radar_channels[y, x] = radar_points[i, 2:-1]


            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'radar': radar_channels}

        
        if self.transform:
            sample = self.transform(sample)

        return sample
    

    def random_crop(self, img, depth, rad, box_pos, radar_gt, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)

        box_pos[:, 0] -= x
        box_pos[:, 2] -= x
        box_pos[:, 1] -= y
        box_pos[:, 3] -= y
        box_pos[box_pos < 0] = 0
        box_pos[:, 0][box_pos[:, 0] >= width] = width
        box_pos[:, 1][box_pos[:, 1] >= height] = height
        
        box_pos[:, 2][box_pos[:, 2] >= width] = width
        box_pos[:, 3][box_pos[:, 3] >= height] = height
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        rad = rad[y:y + height, x:x + width, :]
        radar_gt = radar_gt[y:y + height, x:x + width, :]
        return img, depth, rad, box_pos, radar_gt

    def train_preprocess(self, image, depth_gt, radar, box_pos, radar_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            radar = (radar[:, ::-1, :]).copy()
            radar_gt = (radar_gt[:, ::-1, :]).copy()
            n_height, n_width, _ = image.shape
            for i in range(box_pos.shape[0]):
                if np.count_nonzero(box_pos[i]) ==0:
                    continue
                temp = box_pos[i, 0].copy()
                box_pos[i, 0] = n_width - box_pos[i, 2]
                box_pos[i, 2] = n_width - temp
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt, radar, box_pos, radar_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.image_paths)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal, radar = sample['image'], sample['focal'], sample['radar']
        image = self.to_tensor(image)
        radar = self.to_tensor(radar)
        image = self.normalize(image)

        depth = sample['depth']
        if self.mode == 'train':
            box_pos = sample['box_pos']
            box_pos = self.to_tensor(box_pos)
            depth = self.to_tensor(depth)
            radar_gt = sample['radar_gt']
            radar_gt = self.to_tensor(radar_gt)
            return {'image': image, 'depth': depth, 'focal': focal, 'radar': radar, 'box_pos': box_pos, 'radar_gt': radar_gt}
        else:
            return {'image': image, 'depth': depth, 'focal': focal, 'radar': radar}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            if len(pic.shape) > 2:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
                return img
            else:
                arr = torch.from_numpy(pic)
                return arr

        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
