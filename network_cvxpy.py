import torch
import torch.nn as nn
import math
import numpy as np
import cvxpy as cp
from utils import *


class Unfolding(nn.Module):
    def __init__(self, sigma2 = 1, P_max = 1, Ksize = [3, 1, 2, 1, 1], layer_num = 3):
        super(Unfolding, self).__init__()
        self.layer_num = layer_num
        self.sigma2 = sigma2
        self.features = nn.ModuleList([])
        for i in range(self.layer_num):
            self.features.append( OneStep(sigma2 = sigma2, P_max = P_max, Ksize = Ksize) )

    def forward(self, p, Theta, Tx, Rx, filename = None, save_data = None):
        result = []
        for i in range(self.layer_num):
            print('========================================================')
            print('==================== layer '+str(i)+' ===========================')
            print('========================================================')
            p, Theta = self.features[i](p, Theta, Tx, Rx, num = i)
            loss = objective(p, Theta, Tx, Rx, self.sigma2)
            print('Average capacity in iteration '+ str(i) + ' is '+ str(loss.item()) + '.')
            
            val_batch_size = Theta.shape[0]
            L = Theta.shape[1]
            Ml = Theta.shape[2]
            K = p.shape[1]
            device = Tx.device
            angle = math.pi*2*torch.rand(val_batch_size, L, Ml).to(device)
            snr = 10
            power = math.sqrt(10**(snr/10))
            p2 = power*torch.ones(val_batch_size, K).to(device)
            Theta2 = torch.stack((torch.cos(angle), torch.sin(angle)), dim = 2)
            print('Rand Theta in iteration '+ str(i) + ' is '+ str(objective(p, Theta2, Tx, Rx, self.sigma2).item()) + '.')
            print('Peak power in iteration '+ str(i) + ' is '+ str(objective(p2, Theta, Tx, Rx, self.sigma2).item()) + '.')
            print('Peak power Rand Theta in iteration '+ str(i) + ' is '+ str(objective(p2, Theta2, Tx, Rx, self.sigma2).item()) + '.')
            
            with open(filename, 'a') as file_object:
                file_object.write('Average capacity in iteration '+ str(i) + ' is '+ str(loss.item()) + '.\n')
            save_checkpoint({'Tx':Tx, 'Rx':Rx, 'p':p, 'Theta':Theta}, i, tag='bnps-', path = save_data)
            result.append([p, Theta])
        return result
    
class OneStep(nn.Module):
    def __init__(self, sigma2 = 1, P_max = 1, Ksize = [3, 1, 2, 1, 1]):
        super(OneStep, self).__init__()
        self.sigma2 = sigma2
        self.P_max = P_max
        self.Ksize = Ksize
        # K, L, Ml, M, N
        
    def forward(self, p, Theta, Tx, Rx, num = 1):
        # input_theta: (train_sample_num, L, Ml, 2)
        # K: users; L: number of IRS; Ml: elements of IRS;
        # M: receiver antennas; N: transmitter antennas
        # K, L, Ml, M, Nm = 3, 1, 2, 1, 1
        # Hd : (train_sample_num, K, K, M,  N,  2)      # transmitter-receiver
        # Tx : (train_sample_num, L, K, Ml, N,  2)      # transmitter-IRS
        # Rx : (train_sample_num, K, L, M,  Ml, 2)      # IRS-receiver
        # theta : (train_sample_num, L, L, Ml, Ml, 2)
        device = Tx.device
        train_sample_num = Tx.shape[0]
        K, L, Ml, M, N = self.Ksize
        Tx_matrix = Tx.permute(0, 1, 3, 2, 4, 5).contiguous().view(train_sample_num, L*Ml, K*N, 2)
        Rx_matrix = Rx.permute(0, 1, 3, 2, 4, 5).contiguous().view(train_sample_num, K*M, L*Ml, 2)
        Theta_matrix = Theta.view(train_sample_num, L*Ml, 2)

        Tx_real, Tx_imag = torch.select(Tx_matrix, -1, 0), torch.select(Tx_matrix, -1, 1)
        Rx_real, Rx_imag = torch.select(Rx_matrix, -1, 0), torch.select(Rx_matrix, -1, 1)
        Theta_real, Theta_imag = torch.select(Theta_matrix, -1, 0).diag_embed(), torch.select(Theta_matrix, -1, 1).diag_embed()


        Rx_Theta_real = torch.matmul(Rx_real, Theta_real) - torch.matmul(Rx_imag, Theta_imag)
        Rx_Theta_imag = torch.matmul(Rx_real, Theta_imag) + torch.matmul(Rx_imag, Theta_real)
        h_real = torch.matmul(Rx_Theta_real, Tx_real) - torch.matmul(Rx_Theta_imag, Tx_imag)
        h_imag = torch.matmul(Rx_Theta_real, Tx_imag) + torch.matmul(Rx_Theta_imag, Tx_real)

        h_square = h_real**2 + h_imag**2
        
        h_gain = (p**2).view(train_sample_num, -1, K*N)*h_square
        numerator = h_gain.diagonal(dim1 = 1, dim2 = 2)
        denominator = ( h_gain - numerator.diag_embed() ).sum(dim = 2) + self.sigma2
        mu = numerator / denominator
        
        numerator = torch.sqrt( (1 + mu) * h_gain.diagonal(dim1 = 1, dim2 = 2))
        denominator = h_gain.sum(dim = 2) + self.sigma2
        alpha = numerator / denominator / math.sqrt(2)
        
        numerator = (1 + mu) * alpha**2 * h_square.diagonal(dim1 = 1, dim2 = 2)
        denominator = ( ( (alpha**2).view(train_sample_num, K*N, -1) * h_square ).sum(dim = 1) )**2
        p_out = torch.min(torch.ones_like(p)*self.P_max, torch.sqrt(numerator / (2*denominator)))
        p_out = p_out.detach()
        
        numerator_real = 1/math.sqrt(2) * torch.sqrt(1 + mu)*p_out * h_real.diagonal(dim1 = 1, dim2 = 2)
        numerator_imag = 1/math.sqrt(2) * torch.sqrt(1 + mu)*p_out * h_imag.diagonal(dim1 = 1, dim2 = 2)
        h_gain = (p_out**2).view(train_sample_num, -1, K*N)*h_square
        
        denominator = h_gain.sum(dim = 2) + self.sigma2
        beta_real = numerator_real / denominator
        beta_imag = numerator_imag / denominator
        
        theta_out = torch.zeros(train_sample_num, L, Ml, 2).to(device)
        for sample in range(train_sample_num):
            if sample%100 == 0:
                print('%5d'%sample, end = '')
            theta_out[sample] = self.cvx_opt(cp_p = p_out.cpu().numpy()[sample]**2,
                        cp_Tx_real = Tx_real.cpu().numpy()[sample],
                        cp_Tx_imag = Tx_imag.cpu().numpy()[sample],
                        cp_Rx_real = Rx_real.cpu().numpy()[sample],
                        cp_Rx_imag = Rx_imag.cpu().numpy()[sample],
                        cp_beta_real = beta_real.cpu().numpy()[sample],
                        cp_beta_imag = beta_imag.cpu().numpy()[sample],
                        cp_mu = mu.cpu().numpy()[sample])
        print()
        return p_out, theta_out
    
    def cvx_opt(self, cp_p, cp_Tx_real, cp_Tx_imag, cp_Rx_real, cp_Rx_imag, cp_beta_real, cp_beta_imag, cp_mu):
        K, L, Ml, M, N = self.Ksize
        cp_Theta_real = cp.Variable(L*Ml)
        cp_Theta_imag = cp.Variable(L*Ml)


        cp_Rx_Theta_real = cp_Rx_real@cp.diag(cp_Theta_real) - cp_Rx_imag@cp.diag(cp_Theta_imag)
        cp_Rx_Theta_imag = cp_Rx_real@cp.diag(cp_Theta_imag) + cp_Rx_imag@cp.diag(cp_Theta_real)

        cp_h_real = cp_Rx_Theta_real@cp_Tx_real - cp_Rx_Theta_imag@cp_Tx_imag
        cp_h_imag = cp_Rx_Theta_real@cp_Tx_imag + cp_Rx_Theta_imag@cp_Tx_real

        cp_numerator_real = cp.multiply( cp.diag(cp_h_real),  cp.sqrt(2*cp.multiply(cp_p, 1 + cp_mu)) )
        cp_numerator_imag = cp.multiply( cp.diag(cp_h_imag), cp.sqrt(2*cp.multiply(cp_p, 1 + cp_mu)) )
        cp_h_square = cp.square(cp_h_real) + cp.square(cp_h_imag)
        cp_h_gain = cp.multiply(cp_h_square, cp.vstack([cp_p]*K))
        cp_denominator = cp.sum(cp_h_gain, axis=1) + self.sigma2

        cp_beta_square = cp.square(cp_beta_real) + cp.square(cp_beta_imag)
        objective = cp.Maximize( cp_beta_real@cp_numerator_real + cp_beta_imag@cp_numerator_imag - cp_beta_square@cp_denominator )
        constrains = [cp.square(cp_Theta_real) + cp.square(cp_Theta_imag) <= 1]
        prob = cp.Problem(objective, constrains)
        assert prob.is_dcp(dpp=False)
        result = prob.solve()
        theta_out = torch.stack((torch.from_numpy(cp_Theta_real.value.astype(np.float32)), torch.from_numpy(cp_Theta_imag.value.astype(np.float32))), dim = 1).view(L, Ml, 2)
        return theta_out

def objective(p, Theta, Tx, Rx, sigma2):
    train_sample_num, L, K, Ml, N, _ = Tx.shape
    M = Rx.shape[3]
    Tx_matrix = Tx.permute(0, 1, 3, 2, 4, 5).contiguous().view(train_sample_num, L*Ml, K*N, 2)
    Rx_matrix = Rx.permute(0, 1, 3, 2, 4, 5).contiguous().view(train_sample_num, K*M, L*Ml, 2)
    Theta_matrix = Theta.view(train_sample_num, L*Ml, 2)
#       
    
    Tx_real, Tx_imag = torch.select(Tx_matrix, -1, 0), torch.select(Tx_matrix, -1, 1)
    Rx_real, Rx_imag = torch.select(Rx_matrix, -1, 0), torch.select(Rx_matrix, -1, 1)
    Theta_real, Theta_imag = torch.select(Theta_matrix, -1, 0).diag_embed(), torch.select(Theta_matrix, -1, 1).diag_embed()

    Rx_Theta_real = torch.matmul(Rx_real, Theta_real) - torch.matmul(Rx_imag, Theta_imag)
    Rx_Theta_imag = torch.matmul(Rx_real, Theta_imag) + torch.matmul(Rx_imag, Theta_real)
    h_real = torch.matmul(Rx_Theta_real, Tx_real) - torch.matmul(Rx_Theta_imag, Tx_imag)
    h_imag = torch.matmul(Rx_Theta_real, Tx_imag) + torch.matmul(Rx_Theta_imag, Tx_real)
    
    h_square = h_real**2 + h_imag**2

    h_gain = (p**2).view(train_sample_num, -1, K*N)*h_square
    numerator = h_gain.diagonal(dim1 = 1, dim2 = 2)
    denominator = ( h_gain - numerator.diag_embed() ).sum(dim = 2) + sigma2
    return torch.mean(torch.sum(torch.log(1 + numerator / denominator), dim = 1)/math.log(2))
        
if __name__ == '__main__':
    K, L, Ml, M, N = 5, 2, 10, 1, 1
    layer = Unfolding(sigma2 = 1, P_max = 1, Ksize = [K, L, Ml, M, N], hidden_dim = 20)
    
    Tx = torch.rand(100, L, K, Ml, N, 2)
    Rx = torch.rand(100, K, L, M, Ml, 2)
    p = torch.rand(100, K)
    theta = torch.rand(100, L, Ml, 2)
    result = layer(p, theta, Tx, Rx)
    print(len(result))