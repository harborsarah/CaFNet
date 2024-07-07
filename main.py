# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import time
import argparse
import datetime
import sys
import os
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
from tqdm import tqdm
from models.model import CaFNet
from dataloaders.cafnet_dataloader import *
from models.losses import *

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='CaFNet PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                               type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                         type=str,   help='model name', default='CaFNet')
parser.add_argument('--main_path',                          type=str,   help='main path of data', required=True)
parser.add_argument('--train_image_path',                   type=str,   help='path of training image', required=True)
parser.add_argument('--train_radar_path',                   type=str,   help='path of training radar', required=True)
parser.add_argument('--train_ground_truth_path',            type=str,   help='path of D', required=True)
parser.add_argument('--train_box_pos_path',                 type=str,   help='path of boxes', required=True)
parser.add_argument('--test_image_path',                    type=str,   help='path of testing image', required=True)
parser.add_argument('--test_radar_path',                    type=str,   help='path of testing radar', required=True)
parser.add_argument('--test_ground_truth_path',             type=str,   help='path of testing ground truth', required=True)

parser.add_argument('--encoder',                            type=str,   help='type of image encoder', default='resnet34_bts')
parser.add_argument('--encoder_radar',                      type=str,   help='type of encoder of radar channels', default='resnet18')
parser.add_argument('--radar_input_channels',               type=int,   help='number of input radar channels', default=5)

# Dataset
parser.add_argument('--input_height',                       type=int,   help='input height', default=352)
parser.add_argument('--input_width',                        type=int,   help='input width',  default=704)
parser.add_argument('--max_depth',                          type=float, help='maximum depth in estimation', default=100)
parser.add_argument('--max_dist_correspondence',            type=float, help='Max distance to consider two points correspondence', default=0.5)
parser.add_argument('--patch_size',                         type=int, nargs='+', default=[150, 50], help='Height, width of input patch')
parser.add_argument('--bts_size',                           type=int,   help='initial num_filters in bts decoder', default=512)

# Log and save
parser.add_argument('--log_directory',                      type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',                    type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                           type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                          type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                          help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                           help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                              help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',                       type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                                        help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                           type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                         type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                         type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',                      type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',                  type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',                     type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
parser.add_argument('--reg_loss',                           type=str,   help='loss function for depth regression - l1/silog', default='l1')
parser.add_argument('--w_smoothness',                       type=float, help='Weight of local smoothness loss', default=0.00)

# Multi-gpu training
parser.add_argument('--num_threads',                        type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                         type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                               type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                           type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',                       type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                                type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',                    help='Use multi-processing distributed training to launch '
                                                                             'N processes per node, which has N GPUs. This is the '
                                                                             'fastest way to use PyTorch for either single node or '
                                                                             'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                                 help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--min_depth_eval',                     type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',                     type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eval_freq',                          type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',             type=str,   help='output directory for eval summary,'
                                                                             'if empty outputs to checkpoint folder', default='')





if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from models.bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'mae', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    mae = np.mean(np.abs(gt - pred))

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, mae, d1, d2, d3]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False


def online_eval(model, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(11).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            radar = torch.autograd.Variable(eval_sample_batched['radar'].cuda(args.gpu, non_blocking=True))


            _, _, _, _, pred_depth, _, _ = model(image, radar, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)


        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:-1] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[-1] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[-1].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                            'sq_rel', 'log_rms', 'mae', 'd1', 'd2',
                                                                                            'd3'))
        for i in range(9):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[9]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = CaFNet(args)
    model.train()

    model.decoder.apply(weights_init_xavier)
    model.decoder_radar.apply(weights_init_xavier)

    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(7).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(10, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.encoder.parameters(), 'weight_decay': args.weight_decay},
                                {'params': model.module.encoder_radar1.parameters(), 'weight_decay': args.weight_decay},
                                {'params': model.module.encoder_radar2.parameters(), 'weight_decay': args.weight_decay},
                                {'params': model.module.decoder_radar.parameters(), 'weight_decay': args.weight_decay}, # previously doesn't have this line
                                {'params': model.module.decoder.parameters(), 'weight_decay': 0}],
                                lr=args.learning_rate, eps=args.adam_eps)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = CaFNetDataLoader(args, 'train')
    dataloader_eval = CaFNetDataLoader(args, 'online_eval')


    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    if args.reg_loss == 'silog':
        loss_depth = silog_loss(variance_focus=args.variance_focus)
    elif args.reg_loss == 'l2':
        loss_depth = l2_loss()
    else:
        # default: L1 loss
        loss_depth = l1_loss()

    bcn = binary_cross_entropy()
    smoothness = smoothness_loss_func()

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            radar = torch.autograd.Variable(sample_batched['radar'].cuda(args.gpu, non_blocking=True))
            box_pos = torch.autograd.Variable(sample_batched['box_pos'].cuda(args.gpu, non_blocking=True))
            radar_gt = torch.autograd.Variable(sample_batched['radar_gt'].cuda(args.gpu, non_blocking=True))


            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, rad_confidence, radar_depth = model(image, radar, focal)
            

            mask = depth_gt > 0.01

            loss_d = loss_depth.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss_d_rad = loss_depth.forward(radar_depth, depth_gt, mask.to(torch.bool))

            if args.w_smoothness > 0.00:
                loss_smoothness = smoothness.forward(depth_est, image)
                loss_smoothness = loss_smoothness * args.w_smoothness
            else:
                loss_smoothness = 0.0

            loss_confidence = bcn.forward(rad_confidence, radar_gt, mask.to(torch.bool))

            loss = loss_d + 1.0*loss_confidence + 0.5*loss_d_rad + loss_smoothness
            
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            # torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_grad)
            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if args.w_smoothness == 0.0:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.8f}, depth loss: {:.4f}, radar loss: {:.4f}, radar depth loss: {:.4}'.format(\
                        epoch, step, steps_per_epoch, global_step, current_lr, loss, loss_d, loss_confidence, loss_d_rad))
                else:
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.8f}, depth loss: {:.4f}, smooth loss: {:.4f}, radar loss: {:.4f}, radar depth loss: {:.4}'.format(\
                        epoch, step, steps_per_epoch, global_step, current_lr, loss, loss_d, loss_smoothness, loss_confidence, loss_d_rad))

                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss_depth, global_step)
                    writer.add_scalar('bcn_loss', loss_confidence, global_step)
                    writer.add_scalar('total_loss', loss, global_step)

                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node)
                if eval_measures is not None:
                    for i in range(10):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 7 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 7 and measure > best_eval_measures_higher_better[i-7]:
                            old_best = best_eval_measures_higher_better[i-7].item()
                            best_eval_measures_higher_better[i-7] = measure.item()
                            is_best = True
                        if is_best:
                        # if True:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                set_misc(model)
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()

def main():
    if args.mode != 'train':
        print('main.py is only for training. Use test.py instead.')
        return -1
    runtime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.model_name = runtime + '_' + args.model_name

    model_filename = args.model_name + '.py'
    command = 'mkdir ' + args.log_directory + '/' + args.model_name
    os.system(command)

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
