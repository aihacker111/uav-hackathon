from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import torch
from progress.bar import Bar
from lib.utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # 前向推理, 获取网络网络输出
        if 'pre_input' in batch:
            outputs = self.model.forward(batch['pre_input'], batch['input'])
        else:
            outputs = self.model.forward(batch['input'])

        # 根据网络输出和ground truth计算loss
        loss, loss_stats = self.loss.forward(outputs=outputs, batch=batch)

        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)

        # 是否添加loss对象中的可学习参数到优化器中进行优化
        # eg: MOTLoss中的ReID classifier中的可学习参数
        self.optimizer.add_param_group({'params': self.loss.parameters()})
        # for item in self.loss.parameters():
        #     print(item)

    def set_device(self, gpus, chunk_sizes, device):
        dev_ids = [i for i in range(len(gpus))]
        # dev_ids = [int(x) for x in gpus]
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss,
                                                device_ids=dev_ids,  # device_ids=gpus,
                                                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    # def set_device(self, local_rank, device):
    #     # dev_ids = [i for i in range(len(gpus))]
    #     for state in self.optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(device=device, non_blocking=True)
    #
    #     self.model_with_loss.to(device)
    #     # global device
    #     self.model_with_loss = torch.nn.parallel.DistributedDataParallel(self.model_with_loss,
    #                                                                          device_ids=[local_rank],
    #                                                                          output_device=local_rank,
    #                                                                          find_unused_parameters=True)


    # Train an epoch
    def run_epoch(self, phase, epoch, data_loader):
        """
        :param phase:
        :param epoch:
        :param data_loader:
        :return:
        """
        model_with_loss = self.model_with_loss

        if phase == 'train':
            model_with_loss.train()  # train phase
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module

            model_with_loss.eval()  # test phase
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        end = time.time()

        # Gradient accumulation: effective_batch = batch_size × accum_steps
        # Weights are updated every accum_steps mini-batches (or at the very
        # last batch of the epoch so no gradients are silently dropped).
        accum_steps = max(1, getattr(opt, 'grad_accum', 1))

        # Bar tracks optimizer steps, not mini-batches, so ETA / step count
        # correctly reflects how many weight updates remain in this epoch.
        num_opt_steps = math.ceil(num_iters / accum_steps)
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_opt_steps)
        opt_step = 0   # optimizer-step counter (used for bar display)

        if phase == 'train':
            self.optimizer.zero_grad()   # clear at epoch start

        # train each batch
        # print('Total {} batches in en epoch.'.format(len(data_loader) + 1))
        for batch_i, batch in enumerate(data_loader):
            if batch_i >= num_iters:
                break

            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            # Forward
            output, loss, loss_stats = model_with_loss.forward(batch)

            # Backwards
            loss = loss.mean()

            is_last_iter      = (batch_i + 1) >= num_iters
            is_accum_boundary = (batch_i + 1) % accum_steps == 0
            do_step           = (phase == 'train') and (is_accum_boundary or is_last_iter)

            if phase == 'train':
                # Scale loss so gradient magnitude is independent of accum_steps
                (loss / accum_steps).backward()

                if do_step:
                    self.optimizer.step()   # update weights
                    self.optimizer.zero_grad()  # reset for next window

            batch_time.update(time.time() - end)
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))

            # Advance bar and update suffix only on optimizer steps (or every
            # batch during val where accum_steps == 1).
            if do_step or phase != 'train':
                opt_step += 1
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, opt_step, num_opt_steps, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

                if not opt.hide_data_time:
                    Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                              '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
                if opt.print_iter > 0:
                    if opt_step % opt.print_iter == 0:
                        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                else:
                    bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        # randomly do multi-scaling for dataset every epoch
  # re-assign scale for each batch

        # shuffule the dataset every epoch
        # data_loader.dataset.shuffle()  # re-assign file id for each idx

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.0

        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
