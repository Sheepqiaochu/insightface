import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.utils.data.distributed

from PGD import PGDAttacker
from backbones import get_model
from dataset import DataLoaderX, ImageLoader
from utils.utils_config import get_config
from torchvision import transforms, utils


def trans_save(label, x, root):
    label = label.item()
    image_dir = os.path.join(root, str(label))
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    count = len([lists for lists in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, lists))])
    image_name = str(label) + '_' + str(count) + '.jpg'
    utils.save_image(x, os.path.join(image_dir, image_name))


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

    pgd = PGDAttacker(12, 10, 1, True, norm_type='l-infty', args=args)
    ToPIL = transforms.ToPILImage
    for step, (img, label) in enumerate(image_loader):
        adv_x = pgd.perturb(backbone, img, label)
        trans_save(label, adv_x, args.save_path)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Attack the target model and generate adversarial examples')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='path of the target model')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--model', type=str, default='r34', help='choose the model type')
    parser.add_argument('--data_root', type=str, default='~/datasets/cropped', help='choose the model type')
    parser.add_argument('--save_path', type=str, default='~/datasets/results', help='path to save the images')
    main(parser.parse_args())
