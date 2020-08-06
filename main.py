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

    test_dataset = datasets.ImageFolder(
        args.test_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)

    logging.info('prepare model')

    if args.model.startswith('vgg'):
        model_archi = 'vgg'
        if args.model.endswith('11'):
            model = vgg11(pretrained=True)
        elif args.model.endswith('11_bn'):
            model = vgg11_bn(pretrained=True)
        elif args.model.endswith('13'):
            model = vgg13(pretrained=True)
        elif args.model.endswith('13_bn'):
            model = vgg13_bn(pretrained=True)
        elif args.model.endswith('16'):
            model = vgg16(pretrained=True)
        elif args.model.endswith('16_bn'):
            model = vgg16_bn(pretrained=True)
        elif args.model.endswith('19'):
            model = vgg19(pretrained=True)
        elif args.model.endswith('19_bn'):
            model = vgg19_bn(pretrained=True)
    elif args.model.startswith('resnet'):
        model_archi = 'resnet'
        if args.model.endswith('18'):
            model = resnet18(pretrained=True)
        elif args.model.endswith('34'):
            model = resnet34(pretrained=True)
        elif args.model.endswith('50'):
            model = resnet50(pretrained=True)
        elif args.model.endswith('101'):
            model = resnet101(pretrained=True)
        elif args.model.endswith('152'):
            model = resnet152(pretrained=True)
    else:
        raise ValueError(f"{args.model} is not available")
        
    model.train(False)
    module_list = sa_map.model_flattening(model)
    act_store_model = sa_map.ActivationStoringNet(module_list)
    DTD = sa_map.DTD()
    loss_func = nn.CrossEntropyLoss()

    logging.info('testing with saliency mapping start')

    test_top1 = 0
    test_top5 = 0
    test_count = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            image = Variable(image)
            target = Variable(target)

            module_stack, output = act_store_model(image)
            loss = loss_func(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            test_count += image.size(0)
            test_top1 += acc1[0] * image.size(0)
            test_top1_avg = test_top1 / test_count
            test_top5 += acc5[0] * image.size(0)
            test_top5_avg = test_top5 / test_count

            if i % 20 == 0:
                logging.info('sample saliency map generation')
                saliency_map = DTD(module_stack, output, 1000, model_archi)
                saliency_map = torch.sum(saliency_map, dim=1)
                saliency_map_sample = saliency_map[0].detach().numpy()
                saliency_map_sample = np.maximum(0, saliency_map_sample)*255*args.heatmap_scale
                saliency_map_sample = np.minimum(255, saliency_map_sample)
                saliency_map_sample = np.uint8(saliency_map_sample)
                saliency_heatmap = cv2.applyColorMap(saliency_map_sample, cv2.COLORMAP_BONE)
                if not os.path.exists(args.sample_dir):
                    os.mkdir(args.sample_dir)
                heatmap_name = f"{i}th_sample.png"
                cv2.imwrite(os.path.join(args.sample_dir, heatmap_name), saliency_heatmap)
                sample_origin = image.cpu().data[0]
                origin_name = f"{i}th_origin.png"
                save_image(sample_origin, os.path.join(args.sample_dir, origin_name))

            logging.info((f"Test, step #{i}/{len(test_loader)},, "
                          f"top1 accuracy {test_top1_avg:.3f}, "
                          f"top5 accuracy {test_top5_avg:.3f}, "
                          f"loss {torch.mean(loss):.3f}, "))

    log = '\n'.join([
        f'ImageNet pretrained {args.model} Saliency Mapping',
        f'# Test Result',
        f'- top1 acc: {test_top1_avg:.4f}',
        f'- top5 acc: {test_top5_avg:.4f}'])

    logging.info('test finish')
    logging.info(log)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=None,
                        help='directory path of ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of inference')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--heatmap_scale', type=int, default=5000,
                        help='scale value for heatmap visualization')
    parser.add_argument('--model', type=str, default='vgg16_bn',
                        choices=['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                                 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34',
                                 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--sample_dir', type=str, default='saliency_map_sample',
                        help='directory of saliency map heatmap sample')

    args = parser.parse_args()

    try:
        test(args)
    except Exception:
        logging.exception("Testing is falied")
