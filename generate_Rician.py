import os
import math
import torch
import numpy as np
# import the global Hyperparameters
from global_var import *

# save the generated dataset, defalut in ./datasets
def save_dataset(dataset, path = './datasets', name = 'train.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path+'/'+name)
    torch.save(dataset, filename)
    
def generate_channel(Ksize = [3, 1, 2, 1, 1], channel_type = 'Gau', seed = 2020):
    # K: users; L: IRSs; M1: elements of IRS; M: receiver antennas; N: transmitter antennas
    K, L, Ml, M, N = Ksize
    global train_sample_num
    global val_sample_num
    np.random.seed(seed)

    # data setup
    epsillon = 10
    noise = -170 + 10*math.log(180*1e3)/math.log(10)
    Tx_room_x, Tx_room_y, Tx_rad = 0, 0, 10
    Rx_room_x, Rx_room_y, Rx_rad = 200, 30, 10

    # training dataset
    Hd = np.zeros((train_sample_num, K, K, M, N)) # transmitter-receiver
    Tx = np.zeros((train_sample_num, L, K, Ml, N))    # transmitter-IRS
    Rx = np.zeros((train_sample_num, K, L, M, Ml))    # IRS-receiver
    
    # location
    # transmitter
    LocTx_rad = np.random.rand(K,1)*Tx_rad
    LocTx_theta = np.random.rand(K,1)*2*math.pi
    LocTx = np.concatenate((LocTx_rad*np.cos(LocTx_theta) + Tx_room_x, LocTx_rad*np.sin(LocTx_theta) + Tx_room_y), axis = -1)
    LocTx = LocTx[np.newaxis,np.newaxis,:,:]
    NormalDirTx = np.random.rand(K)*2*math.pi
    NormalDirTx = NormalDirTx[np.newaxis,np.newaxis,:]
    # receiver
    LocRx_rad = np.random.rand(train_sample_num,K,1)*Rx_rad
    LocRx_theta = np.random.rand(train_sample_num,K,1)*2*math.pi
    LocRx = np.concatenate((LocRx_rad*np.cos(LocRx_theta) + Rx_room_x, LocRx_rad*np.sin(LocRx_theta) + Rx_room_y), axis = -1)
    LocRx = LocRx[:,:,np.newaxis,:]
    NormalDirRx = np.random.rand(train_sample_num, K)*2*math.pi
    NormalDirRx = NormalDirRx[:,:,np.newaxis]
    # IRS
    LocIRS = np.array([[200, 0]])
    NormalDirIRS = np.array([3/4*math.pi])

    assert LocTx.shape == (1, 1, K, 2)
    assert LocRx.shape == (train_sample_num, K, 1, 2)
    assert LocIRS.shape == (L, 2)
    assert NormalDirTx.shape == (1, 1, K)
    assert NormalDirRx.shape == (train_sample_num, K, 1)
    assert NormalDirIRS.shape == (L, )

    # steering vector
    path_IRS_Tx = np.expand_dims(LocIRS, axis = 1) - LocTx
    IRS_Tx_dir = np.arctan2(path_IRS_Tx[...,1], path_IRS_Tx[...,0]) # (1, L, K)
    path_Rx_IRS = LocRx - LocIRS
    Rx_IRS_dir = np.arctan2(path_Rx_IRS[...,1], path_Rx_IRS[...,0]) # (sample, K, L)
    Angle_Tx_t = IRS_Tx_dir - NormalDirTx # (1, L, K)
    Angle_IRS_r = IRS_Tx_dir - NormalDirIRS[np.newaxis,:,np.newaxis] # (1, L, K)
    Angle_IRS_t = Rx_IRS_dir - NormalDirIRS[np.newaxis,np.newaxis,:] # (sample, K, L)
    Angle_Rx_r = Rx_IRS_dir - NormalDirRx # (sample, K, L)
    Steer_Tx_t = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_Tx_t), axis=-1)*np.arange(N))
    Steer_IRS_r = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_IRS_r), axis=-1)*np.arange(Ml))
    Steer_IRS_t = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_IRS_t), axis=-1)*np.arange(Ml))
    Steer_Rx_r = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_Rx_r), axis=-1)*np.arange(M))
    
    if channel_type == 'Rician':
        Hd = 1/math.sqrt(2)*(np.random.randn(*Hd.shape) + 1j*np.random.randn(*Hd.shape))
        Tx = math.sqrt(1/(epsillon+1)) * 1/math.sqrt(2)*(np.random.randn(*Tx.shape) + 1j*np.random.randn(*Tx.shape))
        Tx += math.sqrt(epsillon/(epsillon+1)) * np.expand_dims(Steer_IRS_r, axis = -1)@np.conj(np.expand_dims(Steer_Tx_t, axis = -2))
        Rx = math.sqrt(1/(epsillon+1)) * 1/math.sqrt(2)*(np.random.randn(*Rx.shape) + 1j*np.random.randn(*Rx.shape))
        Rx += math.sqrt(epsillon/(epsillon+1)) * np.expand_dims(Steer_Rx_r, axis = -1)@np.conj(np.expand_dims(Steer_IRS_t, axis = -2))
        
        # transmitter-receiver
        Loss_Hd = 32.6 + 36.7*np.log( np.linalg.norm(LocRx - LocTx, axis = 3) )/math.log(10)
        coeff_Hd = np.sqrt( 10**((-noise  -Loss_Hd)/10) )
        Hd *= coeff_Hd[:,:,:,np.newaxis,np.newaxis]
        
        # transmitter-IRS
        Loss_Tx = 35.6 + 22.0*np.log( np.linalg.norm(path_IRS_Tx, axis = 3) )/math.log(10)
        coeff_Tx = np.sqrt( 10**((-noise/2-Loss_Tx)/10) )
        Tx *= coeff_Tx[:,:,:,np.newaxis,np.newaxis]

        # IRS-receiver
        Loss_Rx = 35.6 + 22.0*np.log( np.linalg.norm(path_Rx_IRS, axis = 3) )/math.log(10)
        coeff_Rx = np.sqrt( 10**((-noise/2-Loss_Rx)/10) )
        Rx *= coeff_Rx[:,:,:,np.newaxis,np.newaxis]

    elif channel_type == 'Gau':
        Hd = 1/np.sqrt(2)*(np.random.randn(*Hd.shape) + 1j*np.random.randn(*Hd.shape))
        Tx = 1/np.sqrt(2)*(np.random.randn(*Tx.shape) + 1j*np.random.randn(*Tx.shape))
        Rx = 1/np.sqrt(2)*(np.random.randn(*Rx.shape) + 1j*np.random.randn(*Rx.shape))
    else:
        print('Train: Generation does not work')
        
    train_Hd, train_Tx, train_Rx = Hd.astype(np.complex64), Tx.astype(np.complex64), Rx.astype(np.complex64)

    # validation dataset
    Hd = np.zeros((val_sample_num, K, K, M, N)) # transmitter-receiver
    Tx = np.zeros((val_sample_num, L, K, Ml, N))    # transmitter-IRS
    Rx = np.zeros((val_sample_num, K, L, M, Ml))    # IRS-receiver
    
    # location
    # transmitter the same as train
    # receiver
    LocRx_rad = np.random.rand(val_sample_num,K,1)*Rx_rad
    LocRx_theta = np.random.rand(val_sample_num,K,1)*2*math.pi
    LocRx = np.concatenate((LocRx_rad*np.cos(LocRx_theta) + Rx_room_x, LocRx_rad*np.sin(LocRx_theta) + Rx_room_y), axis = -1)
    LocRx = LocRx[:,:,np.newaxis,:]
    NormalDirRx = np.random.rand(val_sample_num, K)*2*math.pi
    NormalDirRx = NormalDirRx[:,:,np.newaxis]
    # IRS the same as train

    assert LocRx.shape == (val_sample_num, K, 1, 2)
    assert NormalDirRx.shape == (val_sample_num, K, 1)

    # steering vector
    path_IRS_Tx = np.expand_dims(LocIRS, axis = 1) - LocTx
    IRS_Tx_dir = np.arctan2(path_IRS_Tx[...,1], path_IRS_Tx[...,0]) # (1, L, K)
    path_Rx_IRS = LocRx - LocIRS
    Rx_IRS_dir = np.arctan2(path_Rx_IRS[...,1], path_Rx_IRS[...,0]) # (sample, K, L)
    Angle_Tx_t = IRS_Tx_dir - NormalDirTx # (1, L, K)
    Angle_IRS_r = IRS_Tx_dir - NormalDirIRS[np.newaxis,:,np.newaxis] # (1, L, K)
    Angle_IRS_t = Rx_IRS_dir - NormalDirIRS[np.newaxis,np.newaxis,:] # (sample, K, L)
    Angle_Rx_r = Rx_IRS_dir - NormalDirRx # (sample, K, L)
    Steer_Tx_t = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_Tx_t), axis=-1)*np.arange(N))
    Steer_IRS_r = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_IRS_r), axis=-1)*np.arange(Ml))
    Steer_IRS_t = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_IRS_t), axis=-1)*np.arange(Ml))
    Steer_Rx_r = np.exp(1j*math.pi*np.expand_dims(np.sin(Angle_Rx_r), axis=-1)*np.arange(M))
    
    if channel_type == 'Rician':
        Hd = 1/math.sqrt(2)*(np.random.randn(*Hd.shape) + 1j*np.random.randn(*Hd.shape))
        Tx = math.sqrt(1/(epsillon+1)) * 1/math.sqrt(2)*(np.random.randn(*Tx.shape) + 1j*np.random.randn(*Tx.shape))
        Tx += math.sqrt(epsillon/(epsillon+1)) * np.expand_dims(Steer_IRS_r, axis = -1)@np.conj(np.expand_dims(Steer_Tx_t, axis = -2))
        Rx = math.sqrt(1/(epsillon+1)) * 1/math.sqrt(2)*(np.random.randn(*Rx.shape) + 1j*np.random.randn(*Rx.shape))
        Rx += math.sqrt(epsillon/(epsillon+1)) * np.expand_dims(Steer_Rx_r, axis = -1)@np.conj(np.expand_dims(Steer_IRS_t, axis = -2))
        
        # transmitter-receiver
        Loss_Hd = 32.6 + 36.7*np.log( np.linalg.norm(LocRx - LocTx, axis = 3) )/math.log(10)
        coeff_Hd = np.sqrt( 10**((-noise  -Loss_Hd)/10) )
        Hd *= coeff_Hd[:,:,:,np.newaxis,np.newaxis]
        
        # transmitter-IRS
        Loss_Tx = 35.6 + 22.0*np.log( np.linalg.norm(path_IRS_Tx, axis = 3) )/math.log(10)
        coeff_Tx = np.sqrt( 10**((-noise/2-Loss_Tx)/10) )
        Tx *= coeff_Tx[:,:,:,np.newaxis,np.newaxis]        

        # IRS-receiver
        Loss_Rx = 35.6 + 22.0*np.log( np.linalg.norm(path_Rx_IRS, axis = 3) )/math.log(10)
        coeff_Rx = np.sqrt( 10**((-noise/2-Loss_Rx)/10) )
        Rx *= coeff_Rx[:,:,:,np.newaxis,np.newaxis]

    elif channel_type == 'Gau':
        Hd = 1/np.sqrt(2)*(np.random.randn(*Hd.shape) + 1j*np.random.randn(*Hd.shape))
        Tx = 1/np.sqrt(2)*(np.random.randn(*Tx.shape) + 1j*np.random.randn(*Tx.shape))
        Rx = 1/np.sqrt(2)*(np.random.randn(*Rx.shape) + 1j*np.random.randn(*Rx.shape))
    else:
        print('Val: Generation does not work')
        
    val_Hd, val_Tx, val_Rx = Hd.astype(np.complex64), Tx.astype(np.complex64), Rx.astype(np.complex64)

    # save dataset
    save_dataset({'train_Tx': train_Tx, 'train_Rx': train_Rx}, path = './datasets', name = channel_type+'_train.pth.tar')
    save_dataset({'val_Tx': val_Tx, 'val_Rx': val_Rx}, path = './datasets', name = channel_type+'_val.pth.tar')
    print('Datasets are generated successfully!')
    
if __name__ == "__main__":
    generate_channel(Ksize = Ksize, channel_type = 'Rician')