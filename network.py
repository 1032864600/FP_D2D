import torch
import math
import torch.nn as nn

class MyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, upper_bound):

        ctx.save_for_backward(input)
        ctx.upper_bound = upper_bound
        return input.clamp(min=0, max = upper_bound)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input < 0) & (grad_output > 0)] = 0
        grad_input[(input > ctx.upper_bound) & (grad_output < 0)] = 0
        return grad_input, None

class ReLU_threshold(nn.Module):

    def __init__(self, threshold):
        super(ReLU_threshold, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        return torch.clamp(inputs, 0, 1)*self.threshold

class Theta_threshold(nn.Module):

    def __init__(self):
        super(Theta_threshold, self).__init__()

    def forward(self, inputs):
        return torch.clamp(inputs, -1, 1)
    
class Approx(nn.Module):
    def __init__(self, Ksize=[6, 2, 3, 1, 1], hidden_dim=1000, P = 1):
        super(Approx, self).__init__()
        # K, L, Ml, M, N
        self.Ksize = Ksize
        self.inp = 2*2*self.Ksize[0]*self.Ksize[1]*self.Ksize[2]
        self.oup = self.Ksize[0]+self.Ksize[1]*self.Ksize[2]*2
        self.hidden_dim = hidden_dim
        self.P = P
#         self.architecture = nn.Sequential(
#             nn.Linear(self.inp, self.hidden_dim, bias = True),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.ReLU(inplace = True),
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.ReLU(inplace = True),
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.ReLU(inplace = True),
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.ReLU(inplace = True),
#             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim, self.oup, bias = True),
#             nn.BatchNorm1d(self.oup),
#         )
        self.architecture = nn.Sequential(
            nn.Linear(self.inp, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.oup, bias = True),
            nn.BatchNorm1d(self.oup),
        )
#         self.architecture = nn.Sequential(
#             nn.Linear(self.inp, self.hidden_dim, bias = True),
#             nn.ReLU(inplace = True),
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
#             nn.ReLU(inplace = True),
# #             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
#             nn.ReLU(inplace = True),
# #             nn.Dropout(0.3),
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
#             nn.ReLU(inplace = True),
# #             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim, self.oup, bias = True),
#         )
        
        self.last_layer_p = nn.Sequential(
            ReLU_threshold(threshold = self.P))    
#         self.last_layer_p = MyClamp.apply
    
    def forward(self, Tx, Rx):
        train_sample_num = Tx.shape[0]
        x = torch.cat((Tx.contiguous().view(train_sample_num, -1), Rx.contiguous().view(train_sample_num, -1)), 1)
        new_x = self.architecture(x)
        xaxis = new_x[:,self.Ksize[0]:self.Ksize[0]+self.Ksize[1]*self.Ksize[2]]
        yaxis = new_x[:,self.Ksize[0]+self.Ksize[1]*self.Ksize[2]:]
#         cond = torch.sqrt(xaxis**2+yaxis**2)
#         amp = torch.ones_like(cond)
#         amp[cond >= 1] = cond[cond >= 1]
        amp = torch.sqrt(xaxis**2+yaxis**2)
#         return new_x[:,:self.Ksize[0]], torch.stack((xaxis, yaxis), dim = 2).view(train_sample_num, self.Ksize[1], self.Ksize[2], 2)
#         return self.last_layer_p(new_x[:,:self.Ksize[0]], self.P), torch.stack((xaxis/amp, yaxis/amp), dim = 2).view(train_sample_num, self.Ksize[1], self.Ksize[2], 2)
        return self.last_layer_p(new_x[:,:self.Ksize[0]]), torch.stack((xaxis/amp, yaxis/amp), dim = 2).view(train_sample_num, self.Ksize[1], self.Ksize[2], 2)
    
class Loss_func(nn.Module):
    def __init__(self):
        super(Loss_func, self).__init__()
        
    def forward(self, p, Theta, Tx, Rx, sigma2):
        result = objective(p, Theta, Tx, Rx, sigma2)
        return torch.mean(result)

def objective(p, Theta, Tx, Rx, sigma2):
    train_sample_num, L, K, Ml, N, _ = Tx.shape
    M = Rx.shape[3]
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
    
#     Theta_Tx_real = torch.matmul(Theta_real, Tx_real) - torch.matmul(Theta_imag, Tx_imag)
#     Theta_Tx_imag = torch.matmul(Theta_imag, Tx_real) + torch.matmul(Theta_real, Tx_imag)
#     h_real = torch.matmul(Rx_real, Theta_Tx_real) - torch.matmul(Rx_imag, Theta_Tx_imag)
#     h_imag = torch.matmul(Rx_real, Theta_Tx_imag) + torch.matmul(Rx_imag, Theta_Tx_real)
#     return torch.mean(h_real)
#     return torch.mean( Rx_Theta_real ) + torch.mean( Rx_Theta_imag ) + torch.mean( Theta_Tx_real ) + torch.mean( Theta_Tx_imag )
    h_square = h_real**2 + h_imag**2

    h_gain = p.view(train_sample_num, -1, K*N)*h_square
    numerator = h_gain.diagonal(dim1 = 1, dim2 = 2)
    denominator = ( h_gain - numerator.diag_embed() ).sum(dim = 2) + sigma2
    return torch.sum(torch.log(1 + numerator / denominator), dim = 1)/math.log(2)

if __name__ == '__main__':
    K, L, Ml, M, N = 6, 2, 3, 1, 1
    Ksize = [K, L, Ml, M, N]
    sigma2 = 1
    model_Approx = Approx(Ksize=Ksize, hidden_dim=1000, P = 1)
    loss_function = Loss_func()
    
    Tx = torch.zeros(100, L, K, Ml, N, 2) 
    Rx = torch.zeros(100, K, L, M, Ml, 2)
    p, theta = model_Approx(Tx, Rx)
    print('p.shape: ', p.shape, ', theta.shape: ', theta.shape)
    loss = loss_function(p, theta, Tx, Rx, sigma2)
    print(loss)