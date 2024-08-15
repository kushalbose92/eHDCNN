import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.hyp_layers as hyp_layers
import manifolds
from torch.nn.modules.module import Module
import numpy as np 

MAE = nn.L1Loss()

def train(model, optimizer, x_train, y_train):
    
    model.train()
    optimizer.zero_grad()
    total_loss=0
    s,t=x_train.shape
    for i in range(s):
        out = model(x_train[i])
        loss_val= F.mse_loss(out.float(), y_train[i].float())
        total_loss+=loss_val
    total_loss.backward()
    optimizer.step()
    print(f'Train Loss: {loss_val/s: .8f}')
    

def test(model, x_test, y_test):
    model.eval()
    p,q=x_test.shape
    loss=0
    for i in range(p):
        out = model(x_test[i])
        mae = F.mse_loss(out.float(), y_test[i].float())
        loss+=mae
    maee=loss/p
    d=maee
    print(f'Test Loss: {maee: .8f}')
    return d

# create a function (this my favorite choice)
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def test_store(model, x_test, y_test):
    model.eval()
    p,q=x_test.shape
    loss=0
    for i in range(p):
        out = model(x_test[i])
        mae = RMSELoss(out, y_test[i])
        loss+=mae
    maee=loss/p
    d=maee
    return d
    

    

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
    def expansive_conv(self, w, v,i):
        # v_euc = self.manifold.logmap0(v, self.c)
        # v_euc = self.map_hyp_feat_to_euc(v, self.c)[0]
        v_euc = v
        #print("v_euc ", v_euc.shape)
        # performs conv operation
        conv_length = self.input_size + (i+1)*self.filter_length
        total_conv = torch.zeros(conv_length).to(v.device)
        for j in range(conv_length):
            t = 0.0
            for l in range(self.input_size):
                if (j-l) < 0 or (j-l) >= self.filter_length:
                    t += 0
                else:
                    #print(w[j-l], "  ", v_euc[l])
                    t += (w[j-l] * v_euc[l])
                    #print(j+1, " ", l+1, " t ", t.shape)
            total_conv[j] = t
        conv_out = self.map_euc_feat_to_hyp(total_conv, self.c)
        return conv_out
    
    # forward function
    def forward(self, hk):
        hk_prev = hk
        for i in range(self.num_layers):
            expansive_conv_out = self.expansive_conv(self.w_list[i], hk_prev,i)
            out = self.manifold.mobius_add(expansive_conv_out, self.bk_list[i], self.c)
            out = F.relu(out)
            hk_prev = out
        return out 

train_iter = 100
device = 'cpu'
input_dim = 10
#num_samples = 1000
mean = 0.0
std_dev = 0.01

"""
epsilon = torch.normal(mean, std_dev, size=(num_samples,)).to(device)
x = torch.empty(num_samples, input_dim).uniform_(-1, 1).to(device)
norm_tensor=torch.norm(x,dim=1)
y=((torch.sqrt(norm_tensor))/(1+torch.sqrt(norm_tensor))+epsilon).to(device)
y_1=((torch.sqrt(norm_tensor))/(1+torch.sqrt(norm_tensor))).to(device)
"""

df=pd.read_csv(r"C:\Users\SAGAR GHOSH\OneDrive\Desktop\Dissertation\EHDCNN\Simulation\house_price\house_data.csv")
x=df.iloc[:,1:].values
y=df.iloc[:,0].values

x=torch.from_numpy(x)
y=torch.from_numpy(y)

p,q=x.shape
num_samples=p
input_dim=q
#print(x.shape, "  ", y.shape)
# print(x)
# import sys 
# sys.exit()


store=np.zeros((train_iter,2))

manifold_name = 'PoincareBall'
if manifold_name == 'PoincareBall':
    manifold = manifolds.PoincareBall()
c = 0.000000000001
filter_length = 8
input_size = input_dim
num_layers = 4
lr = 0.01
w_decay = 0.0005

train_split = 0.80
test_split = num_samples - train_split
num_train = int(num_samples * train_split)
num_test = num_samples - num_train 
sample_idx = np.arange(0, num_samples, 1)
np.random.shuffle(sample_idx)
train_idx = sample_idx[:num_train]
test_idx = sample_idx[num_test:]
x_train = x[train_idx]
y_train = y[train_idx]
x_test = x[test_idx]
y_test = y[test_idx]

hyp_model = HDCNN(manifold, c, filter_length, input_size, num_layers, device)
optimizer = torch.optim.Adam(hyp_model.parameters(), lr=lr, weight_decay=w_decay)

# train

epoch_number=0
for t in range(train_iter):
    epoch_number+=1
    print("Epoch Number: {}".format(epoch_number))
    train(hyp_model, optimizer, x_train, y_train)
    test(hyp_model, x_test, y_test)
    d=test_store(hyp_model, x_test, y_test)
    store[t,0],store[t,1]=epoch_number, d
    print("\n")


data_df=pd.DataFrame(store, columns=['No of Epochs','Test RMSE Loss'])
data_df.to_csv(r"C:\Users\SAGAR GHOSH\OneDrive\Desktop\Dissertation\EHDCNN\Simulation\Synthetic_exp_dataset\rmse_loss_100.csv")
