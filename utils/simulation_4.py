import torch 
import pandas as pd
import numpy as np 
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.hyp_layers as hyp_layers
import manifolds
from torch.nn.modules.module import Module
import numpy as np 

MAE = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

device='cpu'
def train(model, optimizer, x_train, y_train):
    
    model.train()
    optimizer.zero_grad()
    total_loss=0
    s,t=x_train.shape
    p,q=y_train.shape
    for i in range(s):
        out = model(x_train[i])
        #print(y_train[i].shape)
        """ 
        out_p=np.zeros(q)
        for j in range(q):
            out_p[j]=out[j]
        y_train_1=np.zeros(q)
        a=y_train[i]    
        for j in range(q):
            y_train_1[j]=a[j] """  
        
        loss_val= criterion(out.float(), y_train[i].float())
        total_loss+=loss_val
    #print(type(total_loss), type(out))    
    total_loss.backward()
    optimizer.step()
    print(f'Train Loss: {loss_val/s: .8f}')
    

# def test(model, x_test, y_test):
#     model.eval()
#     p,q=x_test.shape
#     loss=0
#     test_correct = 0
#     for i in range(p):
#         out = model(x_test[i])
#         pred = torch.argmax(out)
#         gt = torch.argmax(y_test[i])
#         if pred == gt:
#             test_correct += 1
#         #print(out.shape)
#         #print(y_test[i].shape)
#         maee = F.mse_loss(out.float(), y_test[i].float())
#         loss+=maee
#     test_corr=test_correct/p    
#     print(f'Test Accuracy:{test_corr: .8f}')
#     maee=loss/p    
#     print(f'Test Loss: {maee: .8f}')
    
def test_acr(model, x_test, y_test):
    model.eval()
    p,q=x_test.shape
    test_correct = 0
    for i in range(p):
        out = model(x_test[i])
        pred = torch.argmax(out)
        gt = torch.argmax(y_test[i])
        if pred == gt:
            test_correct += 1
        #print(out.shape)
        #print(y_test[i].shape)
        #maee = F.mse_loss(out.float(), y_test[i].float())
        #loss+=maee
    test_corr=test_correct/p 
    return test_corr   
    # print(f'Test Accuracy:{test_corr: .8f}')
    # maee=loss/p    
    # print(f'Test Loss: {maee: .8f}')





class HDCNN(nn.Module):
    def __init__(self, manifold, c, filter_length, input_size, num_layers, device, num_classes):
        super(HDCNN, self).__init__()
        self.manifold = manifold
        self.c = c
        self.input_size = input_size
        self.filter_length = filter_length
        self.device = device
        self.num_classes=num_classes
        self.num_layers = num_layers
        self.w_list = nn.ParameterList()
        self.bk_list = nn.ParameterList()
        
        for i in range(self.num_layers):
            w = nn.Parameter(torch.rand(self.filter_length), requires_grad = True).to(self.device)
            bk = nn.Parameter(torch.rand(self.input_size + (i+1) * self.filter_length), requires_grad = True).to(self.device)
            self.w_list.append(w)
            self.bk_list.append(bk)
        self.trans_last = nn.Parameter(torch.rand(num_classes, self.input_size + (self.num_layers * self.filter_length)), requires_grad = True).to(self.device)
    
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
            #print(i)
            expansive_conv_out = self.expansive_conv(self.w_list[i], hk_prev,i)
            out = self.manifold.mobius_add(expansive_conv_out, self.bk_list[i], self.c)
            out = F.relu(out)
            hk_prev = out
            #z=self.map_hyp_feat_to_euc(out, self.c)
            #print(z.shape)
        final_out = self.map_euc_feat_to_hyp((torch.mm(self.trans_last, self.map_hyp_feat_to_euc(out, self.c).unsqueeze(1))), self.c)
        #print(final_out.shape)
        final_out = F.softmax(final_out, dim = 0)
        final_out=final_out.squeeze(1)
        #print(final_out)
        return final_out 



device = 'cpu'

"""

input_dim = 10
num_samples = 100


mean = 0.0
std_dev = 0.01
input_dim=30
"""

"""

epsilon = torch.normal(mean, std_dev, size=(num_samples,)).to(device)
x = torch.empty(num_samples, input_dim).uniform_(-1, 1).to(device)
y = ((torch.sin(torch.norm(x,p=2)) / torch.norm(x,p=2)) + epsilon).to(device)

"""


#print(x.shape, "  ", y.shape)
# print(x)
# import sys 
# sys.exit()

manifold_name = 'PoincareBall'
if manifold_name == 'PoincareBall':
    manifold = manifolds.PoincareBall()
c = 0.00001
filter_length = 9
#input_size = input_dim
num_layers = 4
lr = 0.01
w_decay = 0.0005
train_iter = 100

store=np.zeros((train_iter, 2))
"""
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
"""

train_data_1=pd.read_csv(r"C:\Users\SAGAR GHOSH\OneDrive\Desktop\Dissertation\EHDCNN\Simulation\WISDM\mini_train_data.csv")
test_data_1=pd.read_csv(r"C:\Users\SAGAR GHOSH\OneDrive\Desktop\Dissertation\EHDCNN\Simulation\WISDM\mini_test_data.csv")


"""
df_train=pd.get_dummies(train_data_1.iloc[:,-1], dtype=int)
train_data_1=train_data_1.drop(train_data_1.columns[[-1]], axis=1)
train_data=pd.concat([train_data_1, df_train], axis=1)

df_test=pd.get_dummies(test_data_1.iloc[:,-1], dtype=int)
test_data_1=test_data_1.drop(test_data_1.columns[[-1]], axis=1)
test_data=pd.concat([test_data_1, df_train], axis=1)
"""


x_train_n=train_data_1.iloc[:,6:].values
y_train_n=train_data_1.iloc[:,0:6].values
x_test_n=test_data_1.iloc[:,6:].values
y_test_n=test_data_1.iloc[:,0:6].values

x_train=torch.from_numpy(x_train_n).to(device)
y_train=torch.from_numpy(y_train_n).to(device)
x_test=torch.from_numpy(x_test_n).to(device)
y_test=torch.from_numpy(y_test_n).to(device)
#print(y_train)
p,q=y_train.shape
num_classes=q

"""
train_sample=8000
num_layers=int((train_sample)**(1/4))+1
x_train=torch.empty(train_sample, input_dim).uniform_(-1, 1).to(device)
epsilon_train=torch.normal(mean, std_dev, size=(train_sample,)).to(device)
y_train=((torch.sin(torch.norm(x_train,p=2)) / torch.norm(x_train,p=2))+epsilon_train).to(device)

test_sample=2000
x_test=torch.empty(test_sample, input_dim).uniform_(-1,1).to(device) 
y_test=((torch.sin(torch.norm(x_test,p=2)) / torch.norm(x_test,p=2))).to(device)

"""

input_size=x_train_n.shape[1]
hyp_model = HDCNN(manifold, c, filter_length, input_size, num_layers, device, num_classes)
optimizer = torch.optim.Adam(hyp_model.parameters(), lr=lr, weight_decay=w_decay)

# train
epoch_number=0
for t in range(train_iter):
    epoch_number+=1
    print("Epoch Number: {}".format(epoch_number))
    train(hyp_model, optimizer, x_train, y_train)
    d=test_acr(hyp_model, x_test, y_test)
    print(f'Test Accuracy: {d: .8f}')
    store[t,0], store[t,1]=t,d
    print("\n")


data_df=pd.DataFrame(store, columns=['Number of Epochs', 'Test Accuracy'])
data_df.to_csv(r"C:\Users\SAGAR GHOSH\OneDrive\Desktop\Dissertation\EHDCNN\Simulation\WISDM\test_acr_1.csv")
#test(hyp_model, x_test, y_test)
