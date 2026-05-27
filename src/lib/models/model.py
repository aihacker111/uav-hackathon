from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.deimv2_jde import get_deimv2_jde

_model_factory = {
    'dla': get_dla_dcn,
    'deimv2': get_deimv2_jde,
}

def create_model(arch, heads, head_conv, **kwargs):
    """
    :param arch:      e.g. 'dla_34'  or  'deimv2'
    :param heads:     dict of head_name → output_channels
    :param head_conv: intermediate channels in detection heads
    :param kwargs:    extra kwargs forwarded to the model factory
                      (e.g. deimv2_pretrained='.../model.pth' for DEIMv2)
    :return: nn.Module
    """
    # parse 'dla_34' → arch='dla', num_layers=34
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch_key   = arch[:arch.find('_')] if '_' in arch else arch

    get_model = _model_factory[arch_key]

    if arch_key == 'deimv2':
        # DEIMv2 factory does not take num_layers; pass kwargs directly
        model = get_model(heads=heads, head_conv=head_conv, **kwargs)
    else:
        model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)

    return model


def load_model_pretrain (model,
               model_path,
               optimizer=None,
               resume=False,
               lr=None,
               lr_step=None,
               freeze_params=False):
    """
    Load a model from a checkpoint and optionally freeze only the parameters loaded from the checkpoint.

    Args:
        model: The model to load weights into.
        model_path (str): Path to the checkpoint file.
        optimizer: The optimizer to resume (if provided and resume=True).
        resume (bool): If True, resume training with optimizer state and learning rate.
        lr (float): Initial learning rate for resuming optimizer.
        lr_step (list): List of epochs to decay learning rate.
        freeze_params (bool): If True, freeze only the parameters loaded from the checkpoint.

    Returns:
        model: The loaded model (with checkpoint parameters frozen if freeze_params=True).
        optimizer: The optimizer with resumed state (if provided and resume=True).
        start_epoch: The starting epoch for resumed training.
    """
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'epoch' in checkpoint.keys():
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    if 'state_dict' in checkpoint.keys():
        state_dict_ = checkpoint['state_dict']
    else:
        state_dict_ = checkpoint
    state_dict = {}

    # Convert data_parallel to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # Track parameters loaded from checkpoint
    checkpoint_keys = set(state_dict.keys())

    # Check loaded parameters and created model parameters
    msg = ('If you see this, your model does not fully load the '
           'pre-trained weight. Please make sure '
           'you have correctly specified --arch xxx '
           'or set the correct --num_classes for your own dataset.')
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if k not in state_dict:
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # Freeze only checkpoint-loaded parameters if requested
    if freeze_params:
        for name, param in model.named_parameters():
            if name in checkpoint_keys and param.shape == state_dict[name].shape:
                param.requires_grad = False
        print('Parameters loaded from checkpoint have been frozen.')

    # Resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

def load_model(model,
               model_path,
               optimizer=None,
               resume=False,
               lr=None,
               lr_step=None):
    """
    """
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'epoch' in checkpoint.keys():
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    if 'state_dict' in checkpoint.keys():
        state_dict_ = checkpoint['state_dict']
    else:
        state_dict_ = checkpoint
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            # print('Drop parameter {}.'.format(k) + msg)
            pass
    for k in model_state_dict:
        if not (k in state_dict):
            # print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

#
def save_model(path, epoch, model, optimizer=None):
    """
    :param path:
    :param epoch:
    :param model:
    :param optimizer:
    :return:
    """
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    data = {'epoch': epoch,
            'state_dict': state_dict}

    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path)