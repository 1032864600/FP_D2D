import os
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

    # path loss factor
    Cd = 1e-3   # path loss at the reference distance
    CI = 1e-3*10**(10/3)
    aTR, aIT, aIR = 2.8, 2, 2

    # training dataset
    Hd = np.zeros((train_sample_num, K, K, M, N)) # transmitter-receiver
    Tx = np.zeros((train_sample_num, L, K, Ml, N))    # transmitter-IRS
    Rx = np.zeros((train_sample_num, K, L, M, Ml))    # IRS-receiver
    
    # location
    LocTx   = np.dstack((-np.random.rand(train_sample_num, K)*20 + 20, -np.random.rand(train_sample_num, K)*20 + 20))    # transmitter
    LocRx   = np.dstack((-np.random.rand(train_sample_num, K)*20 + 50, -np.random.rand(train_sample_num, K)*20 + 20))   # receiver
    LocIRS  = np.array([25, 20])    # IRS
    
    if channel_type == 'Ray':
        # transmitter-receiver
        coeff1 = np.sqrt(Cd*np.linalg.norm(LocTx[:,np.newaxis,:,:] - LocRx[:,:,np.newaxis,:], axis = 3)**(-aTR))
        tmpHd = 1/np.sqrt(2)*(np.random.randn(*Hd.shape) + 1j*np.random.randn(*Hd.shape))
        Hd = coeff1[:,:,:,np.newaxis,np.newaxis]*tmpHd

        # transmitter-IRS
        coeff2 = np.sqrt(CI*np.linalg.norm(LocTx - LocIRS, axis = 2)**(-aIT))
        tmpTx = 1/np.sqrt(2)*(np.random.randn(*Tx.shape) + 1j*np.random.randn(*Tx.shape))
        Tx = coeff2[:,:,np.newaxis,np.newaxis]*tmpTx

        # IRS-receiver
        coeff3 = np.sqrt(CI*np.linalg.norm(LocIRS - LocRx, axis = 2)**(-aIR))
        tmpRx = 1/np.sqrt(2)*(np.random.randn(*Rx.shape) + 1j*np.random.randn(*Rx.shape))
        Rx = coeff3[:,:,np.newaxis,np.newaxis]*tmpRx
        
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
    LocTx   = np.dstack((-np.random.rand(val_sample_num, K)*20 + 20, -np.random.rand(val_sample_num, K)*20 + 20))    # transmitter
    LocRx   = np.dstack((-np.random.rand(val_sample_num, K)*20 + 50, -np.random.rand(val_sample_num, K)*20 + 20))   # receiver
    LocIRS  = np.array([25, 20])    # IRS
    if channel_type == 'Ray':
        # transmitter-receiver
        coeff1 = np.sqrt(Cd*np.linalg.norm(LocTx[:,np.newaxis,:,:] - LocRx[:,:,np.newaxis,:], axis = 3)**(-aTR))
        tmpHd = 1/np.sqrt(2)*(np.random.randn(*Hd.shape) + 1j*np.random.randn(*Hd.shape))
        Hd = coeff1[:,:,:,np.newaxis,np.newaxis]*tmpHd

        # transmitter-IRS
        coeff2 = np.sqrt(CI*np.linalg.norm(LocTx - LocIRS, axis = 2)**(-aIT))
        tmpTx = 1/np.sqrt(2)*(np.random.randn(*Tx.shape) + 1j*np.random.randn(*Tx.shape))
        Tx = coeff2[:,:,np.newaxis,np.newaxis]*tmpTx

        # IRS-receiver
        coeff3 = np.sqrt(CI*np.linalg.norm(LocIRS - LocRx, axis = 2)**(-aIR))
        tmpRx = 1/np.sqrt(2)*(np.random.randn(*Rx.shape) + 1j*np.random.randn(*Rx.shape))
        Rx = coeff3[:,:,np.newaxis,np.newaxis]*tmpRx
        
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
    generate_channel(Ksize = Ksize, channel_type = 'Gau')