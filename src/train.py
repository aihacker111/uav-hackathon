from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import random
# my_devs = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = my_devs
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import json
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory

def run(opt):
    torch.manual_seed(opt.seed)
    # np.random.seed(opt.seed)
    # random.seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)  # if opt.task==mot -> JointDataset

    f = open(opt.data_cfg)  # choose which dataset to train '../src/lib/cfg/mot15.json',
    data_config = json.load(f)
    trainset_paths = data_config['train']  # 训练集路径
    dataset_root = data_config['root']  # 数据集所在目录
    print("Dataset root: %s" % dataset_root)
    f.close()

    # Image data transformations
    # ToTensor: HxWxC uint8 [0,255] → CxHxW float32 [0,1]
    # Normalize: re-centre to ImageNet stats that DEIMv2/ViT backbones expect.
    #   mean/std from ImageNet: required when loading DINOv3 / ViT pretrained weights.
    #   If you train from scratch (no pretrained backbone), you can remove Normalize.
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    dataset = Dataset(opt=opt,
                      root=dataset_root,
                      paths=trainset_paths,
                      img_size=opt.input_wh,
                      augment=True,
                      transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 多GPU训练
    print("opt.gpus_str: ", opt.gpus_str)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')  # 设置GPU


    print('Creating model...')
    # For DEIMv2, pass the COCO pretrained path so backbone+encoder are loaded.
    # For all other archs, behaviour is unchanged.
    extra_kwargs = {}
    if opt.arch.startswith('deimv2'):
        if getattr(opt, 'deimv2_pretrained', ''):
            extra_kwargs['deimv2_pretrained'] = opt.deimv2_pretrained
        if getattr(opt, 'vit_weights_path', ''):
            extra_kwargs['vit_weights_path'] = opt.vit_weights_path
        # Pass freeze flag — default False means end-to-end fine-tuning
        extra_kwargs['freeze_backbone'] = getattr(opt, 'freeze_backbone', False)
    model = create_model(opt.arch, opt.heads, opt.head_conv, **extra_kwargs)

    # ---- Optimizer với differential LR (chỉ cho DEIMv2 khi không freeze) ----
    # Backbone+encoder dùng LR nhỏ hơn để bảo vệ pretrained features.
    # Heads dùng full LR để học nhanh các task mới (detection + ReID).
    if (opt.arch.startswith('deimv2')
            and not getattr(opt, 'freeze_backbone', False)
            and getattr(opt, 'backbone_lr_scale', 1.0) != 1.0):
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone.') or name.startswith('encoder.'):
                backbone_params.append(param)
            else:
                head_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': opt.lr * opt.backbone_lr_scale},
            {'params': head_params,     'lr': opt.lr},
        ], lr=opt.lr)
        print(f'Differential LR: backbone×{opt.backbone_lr_scale} ({len(backbone_params)} params), '
              f'heads×1.0 ({len(head_params)} params)')
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    # Get dataloader
    # num_workers: số worker process để load data song song với GPU training.
    # Rule of thumb: 2–4 × số GPU. Dùng 0 khi debug để dễ trace lỗi.
    _nw = 0 if opt.is_debug else min(8, os.cpu_count() or 4)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=_nw,
                                               pin_memory=True,
                                               persistent_workers=(_nw > 0),
                                               drop_last=True)

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        # Train an epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
                           epoch, model, optimizer)
        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print("opt.gpus: ", opt.gpus)
    print('epoch:', opt.num_epochs)
    run(opt)
