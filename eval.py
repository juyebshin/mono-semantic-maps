import os
from datetime import datetime
from argparse import ArgumentParser
from pickletools import uint8
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

# for image save
import torchvision.transforms as T
from PIL import Image
transform = T.ToPILImage()

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders
from src.utils.configs import get_default_configuration, load_config, get_eval_configuration
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
from src.data.nuscenes.utils import STATIC_CLASSES, HDMAPNET_CLASSES
from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise, color_map, map_layer_color, hdmapnet_color

def evaluate(dataloader, model, criterion, summary, config):

    model.eval()

    # Compute prior probability of occupancy
    prior = torch.tensor(config.prior)
    prior_log_odds = torch.log(prior / (1 - prior))

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)
    
    # Iterate over dataset
    for i, batch in enumerate(tqdm(dataloader)):

        # Move tensors to GPU
        if len(config.gpus) > 0:
            batch = [t.cuda() for t in batch]
        
        # Predict class occupancy scores and compute loss
        image, calib, labels, mask, dist = batch
        with torch.no_grad():
            if config.model == 'ved':
                logits, mu, logvar = model(image)
                loss = criterion(logits, labels, mask, mu, logvar)
            else:
                logits = model(image, calib)
                loss = criterion(logits, labels, mask)

        # Update confusion matrix
        scores = logits.cpu().sigmoid()
        confusion.update(scores > config.score_thresh, labels, mask)

        # Update tensorboard
        if i % config.log_interval == 0:
            summary.add_scalar('val/loss', float(loss), i*config.batch_size)
        
        # Visualise
        if i % config.vis_interval == 0:
            visualise(summary, image, scores, labels, mask, dist, i*config.batch_size, 
                      config.train_dataset, split='val')

    # Print and record results
    display_results(confusion, config.train_dataset)
    log_results(confusion, config.train_dataset, summary, 'val', 0)

    return confusion.mean_iou


def visualise(summary, images, scores, labels, masks, distances, step, dataset, split):
    # mask: FOV, occulusion mask

    class_names = STATIC_CLASSES if dataset == 'nuscenes' \
        else HDMAPNET_CLASSES
    map_color = map_layer_color if dataset == 'nuscenes' \
        else hdmapnet_color

    # summary.add_image(split + '/image', image[0], step, dataformats='CHW')
    # summary.add_image(split + '/pred', colorise(scores[0], 'coolwarm', 0, 1),
    #                   step, dataformats='NHWC')
    for idx, (image, score, label, mask, dist) in enumerate(zip(images, scores, labels, masks, distances)): # iterate over batch
        summary.add_image(split + '/image', image, step + idx, dataformats='CHW')
        summary.add_image(split + '/pred', colorise(score, 'coolwarm', 0, 1),
                            step + idx, dataformats='NHWC')
        summary.add_image(split + '/gt', colorise(label, 'coolwarm', 0, 1), 
                            step + idx, dataformats='NHWC')
        summary.add_image(split + '/mask', colorise(mask, 'coolwarm', 0, 1),
                            step + idx, dataformats='HWC')
        summary.add_image(split + '/distance', colorise(dist, 'magma'),
                      step, dataformats='NHWC')
        thres = score > 0.5
        # score[thres] = 0.0
        thres = thres.cpu().data.numpy() # (4, 200, 200)
        out_img = np.zeros((thres.shape[1], thres.shape[2], 3), dtype='uint8') # rgb
        gt_img = np.zeros((thres.shape[1], thres.shape[2], 3), dtype='uint8') # rgb
        for i, (class_name, color) in enumerate(map_color.items()):
            if class_name in class_names:
                out_img[thres[i] & mask.cpu().data.numpy()] = color # thres[i] & ?? ~mask[idx].cpu().data.numpy() &  & (mask[i].cpu().data.numpy().astype(np.bool))
                gt_img[label[i].cpu().data.numpy().astype(np.bool)] = color
        # out_img[~mask[idx].cpu().data.numpy()] = (0, 0, 0)
        summary.add_image(split + '/gt_color', gt_img, step + idx, dataformats='HWC')
        summary.add_image(split + '/pred_color', out_img, step + idx, dataformats='HWC')

        gt_img = np.flipud(gt_img)
        out_img = np.flipud(out_img)
        file_name = '{}'.format(step + idx).zfill(5)
        gt_path = os.path.join(summary.log_dir, 'gt')
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)
        Image.fromarray(gt_img, mode='RGB').save(os.path.join(gt_path, file_name + '.png'))
        pred_path = os.path.join(summary.log_dir, 'pred')
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        Image.fromarray(out_img, mode='RGB').save(os.path.join(pred_path, file_name + '.png'))
        input_path = os.path.join(summary.log_dir, 'image')
        if not os.path.exists(input_path):
            os.mkdir(input_path)
        transform(image).save(os.path.join(input_path, file_name + '.jpg'))


    
    # summary.add_image(split + '/gt', colorise(labels[0], 'coolwarm', 0, 1),
    #                   step, dataformats='NHWC')

    
    # for i, name in enumerate(class_names):
    #     summary.add_image(split + '/pred/' + name, scores[0, i], step, 
    #                       dataformats='HW')
    #     summary.add_image(split + '/gt/' + name, labels[0, i], step, 
    #                       dataformats='HW')
    
    # summary.add_image(split + '/mask', mask[0], step, dataformats='HW')


def display_results(confusion, dataset):

    # Display confusion matrix summary
    class_names = STATIC_CLASSES if dataset == 'nuscenes' \
        else HDMAPNET_CLASSES
    
    print('\nResults:')
    for name, iou_score in zip(class_names, confusion.iou):
        print('{:20s} {:.3f}'.format(name, iou_score)) 
    print('{:20s} {:.3f}'.format('MEAN', confusion.mean_iou))



def log_results(confusion, dataset, summary, split, epoch):

    # Display and record epoch IoU scores
    class_names = STATIC_CLASSES if dataset == 'nuscenes' \
        else HDMAPNET_CLASSES

    for name, iou_score in zip(class_names, confusion.iou):
        summary.add_scalar(f'{split}/iou/{name}', iou_score, epoch)
    summary.add_scalar(f'{split}/iou/MEAN', confusion.mean_iou, epoch)


def load_checkpoint(path, model):
    
    ckpt = torch.load(path)

    # Load model weights
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(ckpt['model'])

    # # Load optimiser state
    # optimizer.load_state_dict(ckpt['optimizer'])

    # # Load scheduler state
    # scheduler.load_state_dict(ckpt['scheduler'])

    return ckpt['epoch'], ckpt['best_iou']


# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    # config = get_default_configuration()
    config = get_eval_configuration()

    # Load dataset options
    config.merge_from_file(f'configs/datasets/{args.dataset}.yml')

    # Load model options
    config.merge_from_file(f'configs/models/{args.model}.yml')

    # Load experiment options
    config.merge_from_file(f'configs/experiments/{args.experiment}.yml')

    # Restore config from an existing experiment
    # if args.eval is not None:
    #     config.merge_from_file(os.path.join(args.eval, 'config.yml'))
    
    # Override with command line options
    config.merge_from_list(args.options)

    # Finalise config
    config.freeze()

    return config


def create_experiment(config, tag, resume=None):

    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
    else:
        # Otherwise, generate a run directory based on the current time
        name = datetime.now().strftime('{}_%y-%m-%d--%H-%M-%S').format(tag)
        logdir = os.path.join(os.path.expandvars(config.logdir), name)
        print("\n==> Creating new experiment in directory:\n" + logdir)
        os.makedirs(logdir)
    
    # Display the config options on-screen
    print(config.dump())
    
    # Save the current config
    with open(os.path.join(logdir, 'config.yml'), 'w') as f:
        f.write(config.dump())
    
    return logdir



def main():

    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, default='eval_hdmapnet_v2_mask',
                        help='optional tag to identify the run')
    parser.add_argument('--dataset', choices=['nuscenes', 'hdmapnet'],
                        default='hdmapnet', help='dataset to train on')
    parser.add_argument('--model', choices=['pyramid', 'vpn', 'ved'],
                        default='pyramid', help='model to train')
    parser.add_argument('--experiment', default='test', 
                        help='name of experiment config to load')
    parser.add_argument('--eval', type=str, default='logs/hdmapnet_v2_3gpus_22-03-30--15-38-58', 
                        help='path to an experiment to evaluate')
    # parser.add_argument('--batch_size', type=int, default=1,
    #                     help='batch size 1 for evaluation')
    parser.add_argument('--options', nargs='*', default=[],
                        help='list of addition config options as key-val pairs')
    args = parser.parse_args()

    # Load configuration
    config = get_configuration(args)
    
    # Create a directory for the experiment
    logdir = create_experiment(config, args.tag)

    # Create tensorboard summary 
    summary = SummaryWriter(logdir)

    # Set default device
    if len(config.gpus) > 0:
        torch.cuda.set_device(config.gpus[0])
    
    # Setup experiment
    model = build_model(config.model, config)
    criterion = build_criterion(config.model, config)
    _, val_loader = build_dataloaders(config.train_dataset, config)

    if args.eval is None:
        raise ValueError("Path to an experiment  not provided")
    
    load_checkpoint(os.path.join(args.eval, 'best.pth'), model)
    
    # Evaluate on the validation set
    val_iou = evaluate(val_loader, model, criterion, summary, config)


if __name__ == '__main__':
    main()