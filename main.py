import torch 
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.hyp_layers as hyp_layers
import manifolds
from torch.nn.modules.module import Module



class HDCNN(nn.Module):
    def __init__(self, manifold, c, filter_length, input_size, num_layers, device):
        super(HDCNN, self).__init__()
        self.manifold = manifold
        self.c = c
        self.input_size = input_size
        self.filter_length = filter_length
        self.device = device
        self.num_layers = num_layers
        self.w_list = nn.ParameterList()
        self.bk_list = nn.ParameterList()
        
        for i in range(self.num_layers):
            w = nn.Parameter(torch.rand(self.filter_length), requires_grad = True).to(self.device)
            bk = nn.Parameter(torch.rand(self.input_size + (i+1) * self.filter_length), requires_grad = True).to(self.device)
            self.w_list.append(w)
            self.bk_list.append(bk)
    
    # mapping from Euclidean space to Hyperbolic space
    def map_euc_feat_to_hyp(self, x, c):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, c), c=c), c=c)
        return x_hyp

    # mapping from Hyperbolic space to Euclidean space
    def map_hyp_feat_to_euc(self, x_hyp, c):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x_hyp, c=c), c =c)
        return x

    # expansive convolution
    def expansive_conv(self, w, v):
        # v_euc = self.manifold.logmap0(v, self.c)
        # v_euc = self.map_hyp_feat_to_euc(v, self.c)[0]
        v_euc = v
        print("v_euc ", v_euc.shape)
        # performs conv operation
        conv_length = self.input_size + self.filter_length
        total_conv = torch.zeros(conv_length).to(v.device)
        for j in range(conv_length):
            t = 0.0
            for l in range(self.input_size):
                if (j-l) < 0 or (j-l) >= self.filter_length:
                    t += 0
                else:
                    print(w[j-l], "  ", v_euc[l])
                    t += (w[j-l] * v_euc[l])
                    print(j+1, " ", l+1, " t ", t.shape)
            total_conv[j] = t
        conv_out = self.map_euc_feat_to_hyp(total_conv, self.c)
        return conv_out
    
    # forward function
    def forward(self, hk):
        hk_prev = hk
        for i in range(self.num_layers):
            expansive_conv_out = self.expansive_conv(self.w_list[i], hk_prev)
            out = self.manifold.mobius_add(expansive_conv_out, self.bk_list[i], self.c)
            out = F.relu(out)
            hk_prev = out
        return out 


device = 'cuda:0'
input_dim = 100
num_samples = 1000
mean = 0.0
std_dev = 0.01
epsilon = torch.normal(mean, std_dev, size=(num_samples,)).to(device)
x = torch.empty(num_samples, input_dim).uniform_(-1, 1).to(device)
y = ((torch.sin(torch.norm(x,p=2)) / torch.norm(x,p=2)) + epsilon).to(device)

print(x.shape, "  ", y.shape)
# print(x)
# import sys 
# sys.exit()

manifold_name = 'PoincareBall'
if manifold_name == 'PoincareBall':
    manifold = manifolds.PoincareBall()
c = 1.0
filter_length = 8
input_size = input_dim
num_layers = 4

hyp_model = HDCNN(manifold, c, filter_length, input_size, num_layers, device)
hyp_out = hyp_model(x)
print(hyp_out.shape)