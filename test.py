from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
from torch.autograd import Variable
from tqdm import tqdm

from dataloaders.cafnet_dataloader import *
from models.model import CaFNet
from PIL import Image

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='CaFNet PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--main_path',                  type=str,       help='main path of data', required=True)
parser.add_argument('--test_image_path',            type=str,       help='path of testing image', required=True)
parser.add_argument('--test_radar_path',            type=str,       help='path of testing radar', required=True)
parser.add_argument('--test_ground_truth_path',     type=str,       help='path of testing ground truth', required=True)
parser.add_argument('--encoder',                    type=str,       help='type of image encoder',default='resnet34_bts')
parser.add_argument('--encoder_radar',              type=str,       help='type of encoder of radar channels', default='resnet34')
parser.add_argument('--radar_input_channels',       type=int,       help='number of input radar channels', default=5)

parser.add_argument('--min_depth_eval',             type=float,     help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',             type=float,     help='maximum depth for evaluation', default=80)
parser.add_argument('--min_depth',                  type=float,     help='minimum depth for training', default=1e-3)
parser.add_argument('--max_depth',                  type=float,     help='maximum depth for training', default=80)
parser.add_argument('--checkpoint_path',            type=str,       help='path to a specific checkpoint to load', default='')
parser.add_argument('--save_lpg',                                   help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size',                   type=int,       help='initial num_filters in bts', default=512)
parser.add_argument('--store_prediction',                           help='if set, store the predicted depth and radar confidence', action='store_true')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

if args.store_prediction:
    save_dir = './eval_result/' + model_dir.split('/')[-1].split('_')[0]
    pred_depth_dir = save_dir + '/pred_depth'
    rad_conf_dir = save_dir + '/rad_conf'
    coarse_depth_dir = save_dir + '/coarse_depth'

    if not os.path.exists(save_dir):
        try:
            os.makedirs(pred_depth_dir)
            os.makedirs(rad_conf_dir)
            os.makedirs(coarse_depth_dir)
        except Exception:
            pass


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    mae = np.mean(np.abs(gt - pred))
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae]

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def test(args):
    args.mode = 'online_eval'
    dataloader = CaFNetDataLoader(args, 'online_eval')

    model = CaFNet(args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))
    eval_measures = torch.zeros(11).cuda()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            radar = Variable(sample['radar'].cuda())
            gt_depth = Variable(sample['depth'].cuda())

            _, _, _, _, pred_depth, rad_confidence, rad_depth = model(image, radar, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            rad_confidence = rad_confidence.cpu().numpy().squeeze()
            rad_depth = rad_depth.cpu().numpy().squeeze()

            if args.store_prediction:
                rad_c = np.uint32(rad_confidence*256.0)
                rad_c = Image.fromarray(rad_c, mode='I')
                rad_c.save(rad_conf_dir + '/' + str(i) + '.png')

                pred_d = np.uint32(pred_depth*256.0)
                pred_d = Image.fromarray(pred_d, mode='I')
                pred_d.save(pred_depth_dir + '/' + str(i) + '.png')

                coarse_d = np.uint32(rad_depth*256.0)
                coarse_d = Image.fromarray(coarse_d, mode='I')
                coarse_d.save(coarse_depth_dir + '/' + str(i) + '.png')

            pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
            pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

            validity_map = np.where(gt_depth > 0, 1, 0)
            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                gt_depth > args.min_depth_eval,
                gt_depth < args.max_depth_eval)
            valid_mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            # silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae
            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            eval_measures[:-1] += torch.tensor(measures).cuda()
            eval_measures[-1] += 1
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[-1].item()
        eval_measures_cpu /= cnt

        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'log10', 'abs_rel', 'sq_rel',
                                                                                            'rmse', 'rmse_log', 'd1', 'd2',
                                                                                            'd3', 'mae'))
        for i in range(9):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[9]))

    return

if __name__ == '__main__':
    args.distributed = False

    test(args)
