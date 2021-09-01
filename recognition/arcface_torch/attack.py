import argparse
import logging
import os

import numpy
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import cv2
from PGD import PGDAttacker
from backbones import get_model
from dataset import DataLoaderX, ImageLoader
from utils.utils_config import get_config
from torchvision import transforms, utils
from dataset import get_dataset
from PIL import Image


def resize(x, index, paths):
    origin_img = Image.open(paths[index][0])
    len_org = origin_img.size[0]
    len_x = 112
    # trans = transforms.Compose(
    #     [transforms.ToPILImage()])
    # x = trans(x)
    cropped = transforms.Compose(
        [transforms.CenterCrop(len_org),
         transforms.ToPILImage])

    if len_org <= len_x:
        x = cropped(x)
        x.save(paths[index][0])
    else:
        edge = len_org / 2
        origin_img.paste(x, (edge, edge))
        cv2.imwrite(paths[index][0], origin_img)
    print("write image:" + paths[index][0])


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

    paths = get_dataset(args.data_root)
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

    pgd = PGDAttacker(12, 10, 2, True, norm_type='l-infty', args=args)

    for step, (img, label, index) in enumerate(image_loader):
        adv_x = pgd.perturb(backbone, img, label)
        # utils.save_image(adv_x, paths[index][0])
        resize(adv_x, index, paths)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Attack the target model and generate adversarial examples')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='path of the target model')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--model', type=str, default='r34', help='choose the model type')
    parser.add_argument('--data_root', type=str, default='~/datasets/cropped', help='choose the model type')
    parser.add_argument('--save_path', type=str, default='/data/users/yangqiancheng/datasets/results',
                        help='path to save the images')
    main(parser.parse_args())
