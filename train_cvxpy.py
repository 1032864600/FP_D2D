import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from network_cvxpy import Unfolding, objective
from utils import *
import math
from global_var import *

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

def get_args():
    parser = argparse.ArgumentParser('DNN_IRS')
#     parser.add_argument('--train-batch-size', type=int, default=int(1000), help='batch size')
#     parser.add_argument('--val-batch-size', type=int, default=int(5e3), help='batch size')
    parser.add_argument('--total-iters', type=int, default=int(2e6), help='total iters')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='init learning rate')
    parser.add_argument('--save-path', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--show-interval', type=int, default=int(50), help='display interval')
    parser.add_argument('--save-interval', type=int, default=int(1e3), help='save interval')
    parser.add_argument('--power-dB', type=float, default=0, help='power constrain in dB')
    parser.add_argument('--gpu', type=str, default='0,', help='gpu_index')
    parser.add_argument('--inherit', type=str, default=None, help='pre-train')
    
    parser.add_argument('--layer-num', type=int, default=20, help='layer-num')
    parser.add_argument('--sigma2', type=float, default=1, help='sigma2')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')

    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
#     train_batch_size = args.train_batch_size
#     val_batch_size = args.val_batch_size
    train_batch_size = train_batch
    val_batch_size = val_batch
    save_path = args.save_path
    power = math.sqrt(10**(args.power_dB/10))
    layer_num = args.layer_num
    sigma2 = args.sigma2
    gamma = args.gamma
    filename = save_path+'.txt'
    
#     t1 = time.time()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    num_workers = 0
    
    K, L, Ml, M, N = Ksize
    
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
    
    model_Approx = Unfolding(sigma2 = sigma2, P_max = power, Ksize = Ksize, layer_num = layer_num)
    
#     criterion = Loss_func()
    
    if use_gpu:
        model_Approx = nn.DataParallel(model_Approx)
#         loss_function = criterion.cuda()
        device = torch.device('cuda')
    else:
#         loss_function = criterion
        device = torch.device('cpu')

    model_Approx = model_Approx.to(device)

    model_Approx.eval()
    with torch.no_grad():
        Tx, Rx = val_channel_dataprovider.next()
        Tx, Rx = Tx.to(device), Rx.to(device)
        angle = math.pi*2*torch.rand(Tx.shape[0], L, Ml).to(device)
        p = power*torch.ones(Tx.shape[0], K).to(device)
        Theta = torch.stack((torch.cos(angle), torch.sin(angle)), dim = 2)
        
#         if True:
#             train_path, lastest_iters = get_lastest_model(path = './save_it2_10')
#             assert train_path is not None
#             iters = lastest_iters
#             train_dataset = torch.load(train_path, map_location=None if use_gpu else 'cpu')
# #             Tx = train_dataset['Tx']
# #             Rx = train_dataset['Rx']
#             p = train_dataset['p']**2
#             Theta = train_dataset['Theta']
#             print('load from checkpoint with iters: ', iters)
#             assert list(train_Tx.shape[1:]) == [L, K, Ml,N, 2]
#             assert list(train_Rx.shape[1:]) == [K, L, M,Ml, 2]

        result = model_Approx(p, Theta, Tx, Rx, filename = filename, save_data = 'save_it3_'+str(int(args.power_dB)))
        
        for i in range(layer_num):
            p, Theta = result[i]
            loss = objective(p, Theta, Tx, Rx, sigma2)
            print('Average capacity in iteration '+ str(i) + ' is '+ str(loss.item()) + '.')
    
if __name__ == '__main__':
    main()