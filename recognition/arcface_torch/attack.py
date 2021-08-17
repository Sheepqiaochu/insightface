import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model
from dataset import MXFaceDataset, SyntheticDataset, DataLoaderX, ImageLoader
from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging

import foolbox as fb
from PGD import PGDAttacker


def main(args):
    cfg = get_config(args.config)
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)

    local_rank = args.local_rank
    torch.cuda.set_device('cuda:' + str(local_rank))

    image_set = ImageLoader(root_dir=args.data_root, local_rank=args.local_rank)
    image_sampler = torch.utils.data.distributed.DistributedSampler(image_set, shuffle=False)
    image_loader = DataLoaderX(
        local_rank=local_rank, dataset=image_set, batch_size=cfg.batch_size,
        sampler=image_sampler, num_workers=2, pin_memory=True, drop_last=True)
    # load model
    backbone = get_model(args.model, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
    try:
        backbone.load_state_dict(torch.load(args.model_path, map_location=torch.device(local_rank)))
        if rank == 0:
            logging.info("backbone loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        if rank == 0:
            logging.info("resume fail, target model doesn't exist")

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])

    train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    image_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=image_sampler, num_workers=2, pin_memory=True, drop_last=True)

    margin_softmax = losses.get_loss(cfg.loss)
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    num_image = len(train_set)
    total_batch_size = cfg.batch_size * world_size
    loss = AverageMeter()

    attack = fb.attacks.LinfPGD
    pgd = PGDAttacker(12, 10, 1, True, norm_type='l-infty', args=args)
    for step, (img, label) in enumerate(image_loader):
       adv_x = pgd.perturb(backbone, img, label)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Attack the target model and generate adversarial examples')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='path of the target model')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--model', type=str, default='r34', help='choose the model type')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--data_root', type=str, default='~/datasets/cropped', help='choose the model type')

    main(parser.parse_args())
