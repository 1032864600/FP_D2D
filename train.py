import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from network import Approx, Loss_func
from utils import *
from global_var import *
import math

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]
    
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.01)
        
def weights_init_to_0(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.constant_(m.weight, 0.0)
        
def get_args():
    parser = argparse.ArgumentParser('DNN_IRS')
#     parser.add_argument('--train-batch-size', type=int, default=int(1e4), help='batch size')
#     parser.add_argument('--val-batch-size', type=int, default=int(5e2), help='batch size')
    parser.add_argument('--total-iters', type=int, default=int(2e6), help='total iters')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--save-path', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--show-interval', type=int, default=int(5e2), help='display interval')
    parser.add_argument('--save-interval', type=int, default=int(4e3), help='save interval')
    parser.add_argument('--power-dB', type=float, default=0, help='power constrain in dB')
    parser.add_argument('--gpu', type=str, default='0,', help='gpu_index')
    parser.add_argument('--inherit', type=str, default=None, help='pre-train')

    parser.add_argument('--sigma2', type=float, default=1, help='sigma2')
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
#     train_batch_size = args.train_batch_size
#     val_batch_size = args.val_batch_size
    train_batch_size = train_batch
    val_batch_size = val_batch
    total_iters = int(args.total_iters)
    learning_rate = args.learning_rate
    save_path = args.save_path
    auto_continue = args.auto_continue
    show_interval = args.show_interval
    save_interval = args.save_interval
    power = 10**(args.power_dB/10)
    inherit = args.inherit
    
    sigma2 = args.sigma2
    
#     t1 = time.time()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    num_workers = 0

    # DataSetup
    K, L, Ml, M, N = Ksize
    hidden_dim =300

    train_path = get_dataset(path = './datasets', name = 'Rician_train.pth.tar')
    train_dataset = torch.load(train_path, map_location=None if use_gpu else 'cpu')

    train_Tx = torch.from_numpy(np.stack((np.real(train_dataset['train_Tx']), np.imag(train_dataset['train_Tx'])), axis = -1))
    train_Rx = torch.from_numpy(np.stack((np.real(train_dataset['train_Rx']), np.imag(train_dataset['train_Rx'])), axis = -1))
    assert list(train_Tx.shape[1:]) == [L, K, Ml,N, 2]
    assert list(train_Rx.shape[1:]) == [K, L, M,Ml, 2]
    train_channel = tuple(zip(train_Tx, train_Rx))
    train_channel_loader = torch.utils.data.DataLoader(
        train_channel, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    train_channel_dataprovider = DataIterator(train_channel_loader)

    val_path = get_dataset(path = './datasets', name = 'Rician_val.pth.tar')
    val_dataset = torch.load(val_path, map_location=None if use_gpu else 'cpu')
    val_Tx = torch.from_numpy(np.stack((np.real(val_dataset['val_Tx']), np.imag(val_dataset['val_Tx'])), axis = -1))
    val_Rx = torch.from_numpy(np.stack((np.real(val_dataset['val_Rx']), np.imag(val_dataset['val_Rx'])), axis = -1))
    assert list(val_Tx.shape[1:]) == [L, K, Ml,N, 2]
    assert list(val_Rx.shape[1:]) == [K, L, M,Ml, 2]
    val_channel = tuple(zip(val_Tx, val_Rx))
    val_channel_loader = torch.utils.data.DataLoader(
        val_channel, batch_size=val_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    val_channel_dataprovider = DataIterator(val_channel_loader)
    print('Data Generating Finished!')
    
    model_Approx = Approx(Ksize=Ksize, hidden_dim=hidden_dim, P = power)
    model_Approx.apply(weights_init)
    optimizer_Approx = torch.optim.Adam(model_Approx.parameters(), lr = learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer_Approx,int(5e3),gamma=0.5,last_epoch=-1)
    
    criterion = Loss_func()

    if use_gpu:
        model_Approx = nn.DataParallel(model_Approx)
        loss_function = criterion.cuda()
        device = torch.device('cuda')
    else:
        loss_function = criterion
        device = torch.device('cpu')

    model_Approx = model_Approx.to(device)

    iters = 0
    y1 = np.zeros(total_iters)
    y2 = np.zeros(total_iters)
    if auto_continue:
        lastest_model, lastest_iters = get_lastest_model(path = save_path)
        if lastest_model is not None:
            iters = lastest_iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            y1[0:iters] = checkpoint['val_cost'][0:iters]
            y2[0:iters] = checkpoint['train_cost'][0:iters]
            model_Approx.load_state_dict(checkpoint['state_approx_dict'], strict=True)
            print('load from checkpoint with iters: ', iters)
#             for i in range(iters):
#                 scheduler.step()

    if iters == 0 and args.inherit != None:
        lastest_model, lastest_iters = get_lastest_model(path = args.inherit)
        if lastest_model is not None:
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model_Approx.load_state_dict(checkpoint['state_approx_dict'], strict=True)
            print('Inherit model from', args.inherit)
        
#     t2 = time.time()
#     print('prepare time: ', t2 - t1)
    while iters < total_iters:
#         t3 = time.time()
        iters += 1
#         scheduler.step()
        model_Approx.train()
        Tx, Rx = train_channel_dataprovider.next()
        Tx, Rx = Tx.to(device), Rx.to(device)
        p, Theta = model_Approx(Tx, Rx)
#         p.retain_grad()
#         Theta.retain_grad()
        loss = -loss_function(p, Theta, Tx, Rx, sigma2)
#         loss = -torch.mean(p) - torch.mean(Theta)
        optimizer_Approx.zero_grad()
        loss.backward()
        optimizer_Approx.step()
        y2[iters - 1] = -loss.item()
        
#         t4 = time.time()
#         print('train time: ', t4 - t3)

        model_Approx.eval()
        with torch.no_grad():
            Tx, Rx = val_channel_dataprovider.next()
            Tx, Rx = Tx.to(device), Rx.to(device)
            p, Theta = model_Approx(Tx, Rx)
            loss = -loss_function(p, Theta, Tx, Rx, sigma2)
#             loss = -torch.mean(p) - torch.mean(Theta)
            y1[iters - 1] = -loss.item()
            
            angle = math.pi*2*torch.rand(val_batch_size, L, Ml).to(device)
            p2 = power*torch.ones(val_batch_size, K).to(device)
            Theta2 = torch.stack((torch.cos(angle), torch.sin(angle)), dim = 2)
            
            
#         print('eval time: ', time.time() - t5)

        if iters % show_interval == 0:
#             print(p[0], 'lr: ', optimizer_Approx.state_dict()['param_groups'][0]['lr'])
#             print('Normal in iteration '+ ' is '+ str(loss_function(p, Theta, Tx, Rx, sigma2).item()) + '.')
#             print('Rand Theta in iteration ' + ' is '+ str(loss_function(p, Theta2, Tx, Rx, sigma2).item()) + '.')
#             print('Peak power in iteration ' + ' is '+ str(loss_function(p2, Theta, Tx, Rx, sigma2).item()) + '.')
#             print('Peak power Rand Theta in iteration ' + ' is '+ str(loss_function(p2, Theta2, Tx, Rx, sigma2).item()) + '.')
#             print(Theta[0,0,1:10])
#             print(iters, ' val: ', y1[iters - 1], ', train: ', y2[iters - 1], scheduler.get_lr()[0])
            print(iters, ' val: ', y1[iters - 1], ', train: ', y2[iters - 1], 'lr: ', optimizer_Approx.state_dict()['param_groups'][0]['lr'])

        if iters % save_interval == 0:
                save_checkpoint({'state_approx_dict': model_Approx.state_dict(), 'val_cost': y1[0: iters], 'train_cost': y2[0: iters]}, iters, tag='bnps-', path = save_path)
    save_checkpoint({'state_approx_dict': model_Approx.state_dict(), 'val_cost': y1, 'train_cost': y2}, total_iters, tag='bnps-', path = save_path)

if __name__ == '__main__':
    main()