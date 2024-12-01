import torch
import torch.nn as nn
import torchvision
from AMSIN import InvDDNet, HaarDownsampling
import numpy as np
from log import TensorBoardX
from utils import *
import train_config as config
from data import Dataset, TestDataset
from tqdm import tqdm
from time import time
import copy
import sys
from pytorch_msssim import ssim
import lpips

log10 = np.log(10)
MAX_DIFF = 1

haar3 = HaarDownsampling(3).cuda()
haar2 = HaarDownsampling(2).cuda()
pus = nn.PixelUnshuffle(2)


def compute_loss(out, batch, out2, out3, rev=False):
    assert out.shape[0] == batch['label256'].shape[0]

    loss = 0
    l1_loss = 0
    psnr = 0
    if rev:
        loss += mse(out[:, 3:], batch['img256'])
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        loss = loss  # 5
        de = batch['img256']
        r_3 = torch.cat([de[:, :1], de[:, :1], de[:, :1]], dim=1)
        loss += mse(out[:, :3], r_3)
    else:
        loss += mse(out[:, 3:], batch['label256'])
        psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
        loss = loss * 10
        ssim_db = out[:, 3:]
        ssim_label = batch['label256']
        ssim_loss = 1 - ssim(ssim_db, ssim_label, data_range=1, size_average=True)
        loss += 2 * ssim_loss
        r = batch['label256'][:, :1]
        g = batch['label256'][:, 1:2]
        b = batch['label256'][:, 2:]
        r_b = pus(r)
        g_b = pus(g)
        b_b = pus(b)
        gt_c = haar3(haar3(batch['label256'])[:, :3])
        loss += 5 * mse(out2[:, :4], r_b)
        loss += 2 * mse(out3[:, 12:], gt_c[:, :3])

    return {'mse': loss, 'psnr': psnr}


def backward(loss, optimizer):
    loss['mse'].backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_(net.module.convlstm.parameters(), 3)

    return


def set_learning_rate(optimizer, epoch):
    optimizer.param_groups[0]['lr'] = config.train['learning_rate']  # * 0.3 ** (epoch // 500)


if __name__ == "__main__":
    tb = TensorBoardX(config_filename='train_config.py', sub_dir=config.train['sub_dir'])
    log_file = open('{}/{}'.format(tb.path, 'train.log'), 'w')

    train_dataset = Dataset('data/path for train', crop_size=(128, 128), mode='train')
    val_dataset = TestDataset('data/path for val', crop_size=(128, 128))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'], shuffle=True,
                                                   drop_last=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train['val_batch_size'], shuffle=True,
                                                 drop_last=True, num_workers=4, pin_memory=True)

    mse = torch.nn.MSELoss().cuda()
    l1 = torch.nn.L1Loss().cuda()
    net = torch.nn.DataParallel(InvDDNet()).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
    # net = SRNDeblurNet(xavier_init_all = config.net['xavier_init_all']).cuda()

    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    last_epoch = -1

    if config.train['resume'] is not None:
        last_epoch = load_model(net, config.train['resume'], epoch=config.train['resume_epoch'])

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer(optimizer, net, config.train['resume_optimizer'], epoch=config.train['resume_epoch'])
        assert last_epoch == _

    # train_loss_epoch_list = []

    train_loss_log_list = []
    val_loss_log_list = []
    first_val = True

    t = time()
    # convlstm_params = net.module.convlstm.parameters()
    # net_params = net.module.parameters()
    best_val_psnr = 0
    best_net = None
    best_optimizer = None

    for epoch in tqdm(range(last_epoch + 1, config.train['num_epochs']), file=sys.stdout):
        set_learning_rate(optimizer, epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader), 'train')
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=sys.stdout,
                                desc='training'):
            t_list = []
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
                batch[k].requires_grad = False

            optimizer.zero_grad()
            '''restore'''
            t = time()
            de = batch['img256']
            x1 = torch.cat([de[:, :1], de[:, :1], de[:, :1]], dim=1)
            out, out2, out3, tt = net(x1, batch['img256'])
            loss = compute_loss(torch.clamp(out, min=1e-5, max=1.0), batch, out2, out3)
            backward(loss, optimizer)
            temp = loss
            '''degrade'''
            gt = batch['label256']
            # re, _ = net(out[:, :3], gt, rev=True)
            # loss = compute_loss(re, batch, re, re, rev=True)
            # backward(loss, optimizer)

            optimizer.step()
            print(tt - t, end='')
            loss = temp

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            train_loss_log_list.append({k: loss[k] for k in loss})
            for k, v in loss.items():
                tb.add_scalar(k, v, epoch * len(train_dataloader) + step, 'train')

        # validate and log
        if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] - 1:
            with torch.no_grad():
                first_val = False
                for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                        desc='validating'):
                    for k in batch:
                        batch[k] = batch[k].cuda(non_blocking=True)
                        batch[k].requires_grad = False
                    de = batch['img256']
                    x1 = torch.cat([de[:, :1], de[:, :1], de[:, :1]], dim=1)
                    out, out2, out3, _ = net(x1, batch['img256'])
                    loss = compute_loss(torch.clamp(out, min=1e-5, max=1.0), batch, out2, out3)
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    val_loss_log_list.append({k: loss[k] for k in loss})

                train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                                       train_loss_log_list[0]}
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}
                for k, v in val_loss_log_dict.items():
                    tb.add_scalar(k, v, (epoch + 1) * len(train_dataloader), 'val')
                if best_val_psnr < val_loss_log_dict['psnr'] + 0.3:
                    best_val_psnr = val_loss_log_dict['psnr']
                    # 验证集较小导致结果有一定的误差，所以只要相差在0.3以内的都作为best保存下来，方便选择最佳效果的模型参数
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)
                if epoch % 50 == 0:
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)

                train_loss_log_list.clear()
                val_loss_log_list.clear()

                tt = time()
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
                        config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
                    val_dataloader) * config.train['val_batch_size']) / (tt - t))

                log_msg += " | train : "
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | val : "
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                log_file.write(log_msg + '\n')
                log_file.flush()
                t = time()
                # print( torch.max( predicts , 1  )[1][:5] )

            # train_loss_epoch_list = []
