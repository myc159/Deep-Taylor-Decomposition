#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os
import pdb
import random
from PIL import Image

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import *
from model import saliency_mapping as sa_map
from torch.autograd import Variable


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(args):

    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("INFO")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info('data loading')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = datasets.ImageFolder(
        args.test_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)

    logging.info('prepare model')

    if args.resnet_model == 'resnet18':
        model = resnet18(pretrained=True)
    elif args.resnet_model == 'resnet34':
        model = resnet34(pretrained=True)
    elif args.resnet_model == 'resnet50':
        model = resnet50(pretrained=True)
    elif args.resnet_model == 'resnet101':
        model = resnet101(pretrained=True)
    elif args.resnet_model == 'resnet152':
        model = resnet152(pretrained=True)
    else:
        raise ValueError(f"{args.resnet_model} is not available")
    model.train(False)
    module_list = sa_map.model_flattening(model)
    act_store_model = sa_map.ActivationStoringResNet(module_list)
    DTD = sa_map.DTD()
    DTD.train(False)

    logging.info('testing with saliency mapping start')

    test_top1 = 0
    test_top5 = 0
    test_count = 0
    act_store_model.train(False)
    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            image = Variable(image)
            target = Variable(target)

            module_stack, output = act_store_model(image)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            test_count += image.size(0)
            test_top1 += acc1[0] * image.size(0)
            test_top1_avg = test_top1 / test_count
            test_top5 += acc5[0] * image.size(0)
            test_top5_avg = test_top5 / test_count

            if i % 10 == 0:
                logging.info('sample saliency map generation')
                saliency_map = DTD(module_stack, output, 1000)
                saliency_map = torch.sum(saliency_map, dim=1)
                saliency_map_sample = saliency_map[0].detach().numpy()
                saliency_map_sample = np.uint8(saliency_map_sample * 255)
                #min_max_range = np.amax(saliency_map_sample) - np.amin(saliency_map_sample)
                #saliency_map_sample = np.uint8(((saliency_map_sample - np.amin(saliency_map_sample)) \
                #    / min_max_range) * 255)
                saliency_heatmap = cv2.applyColorMap(saliency_map_sample, cv2.COLORMAP_HOT)
                if not os.path.exists(args.sample_dir):
                    os.mkdir(args.sample_dir)
                heatmap_name = f"{i}th_sample.png"
                cv2.imwrite(os.path.join(args.sample_dir, heatmap_name), saliency_heatmap)
                sample_origin = image.cpu().data[0]
                origin_name = f"{i}th_origin.png"
                save_image(sample_origin, os.path.join(args.sample_dir, origin_name))

            logging.info((f"Test, step #{i}/{len(test_loader)},, "
                          f"top1 accuracy {test_top1_avg:.3f}, "
                          f"top5 accuracy {test_top5_avg:.3f}"))

    log = '\n'.join([
        f'ImageNet pretrained ResNet Saliency Mapping',
        f'# Test Result',
        f'- top1 acc: {test_top1_avg:.4f}',
        f'- top5 acc: {test_top5_avg:.4f}'])

    logging.info('test finish')
    logging.info(log)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default= 1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--resnet_model', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'])
    parser.add_argument('--sample_dir', type=str, default='saliency_map_sample')

    args = parser.parse_args()

    try:
        test(args)
    except Exception:
        logging.exception("Testing is falied")
