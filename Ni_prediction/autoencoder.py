from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
from torchvision import datasets, transforms  
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from optimizer import Ranger
from dataload2 import keyword_list, price_list

batch_size = 16
epoch = 1200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train_loader = price_list()
train_loader = keyword_list()
train_loader = (train_loader-train_loader.min())/(train_loader.max()-train_loader.min())



class auto_encoder_FFN1(nn.Module):
    def __init__(self,task):
        super(auto_encoder_FFN1, self).__init__()

        if task == 'price':
            input_size = 30*45
            output_size = 10*30
            para = [900, 600]

        elif task == 'keyword':
            input_size = 31*38
            output_size = 10*20
            para = [600, 300]

        self.encoder = nn.Sequential(
            nn.Linear(input_size, para[0]),
            nn.Tanh(),
            nn.Linear(para[0], para[1]),
            nn.Tanh(),
            nn.Linear(para[1], output_size)
            )

        self.decoder = nn.Sequential(
            nn.Linear(output_size,para[1]),
            nn.Tanh(),
            nn.Linear(para[1], para[0]),
            nn.Tanh(),
            nn.Linear(para[0], input_size),
            nn.Sigmoid()
            )

    

    
    def forward(self, x):
        feature = self.encoder(x)
        decoded = self.decoder(feature)
        #print(decoded.shape)
        return decoded, feature


class auto_encoder_FFN2(nn.Module):
    def __init__(self,task):
        super(auto_encoder_FFN2, self).__init__()

        if task == 'price':
            input_size = 45
            output_size = 30
            para = [36,30]

        elif task == 'keyword':
            input_size = 38
            output_size = 20
            para = [30,20]

        self.encoder = nn.Sequential(
            nn.Linear(input_size, para[0]),
            nn.Tanh(),
            nn.Linear(para[0], para[1]),
            nn.Tanh(),
            nn.Linear(para[1], output_size)
            )

        self.decoder = nn.Sequential(
            nn.Linear(output_size,para[1]),
            nn.Tanh(),
            nn.Linear(para[1], para[0]),
            nn.Tanh(),
            nn.Linear(para[0], input_size),
            nn.Sigmoid()
            )

    

    
    def forward(self, x):
        feature = self.encoder(x)
        decoded = self.decoder(feature)
        #print(decoded.shape)
        return decoded, feature



#task = 'price'
task = 'keyword'
model = auto_encoder_FFN1(task).to(device)
optimizer = Ranger(model.parameters())

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')

    return BCE 

def save_model(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_'+str(state['epoch']+1)+'.pth')
    torch.save(state,filename)

def load_model(Net, optimizer, model_file,log_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    print_with_write(log_file,'load from '+model_file)
    checkpoint = torch.load(model_file)
    Net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']+1
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch

def train(epoch, task):
    model.train()
    train_loss = 0
    print(task)
    if task == 'price':
        input_size = 30*45
            

    if task == 'keyword':
        input_size = 31*38
            
    for batch_idx, data in enumerate(train_loader):
        data = data.reshape(-1, input_size)
        #print(data.shape, data.max(), data.min())
        data = torch.from_numpy(data).float().to(device)
        optimizer.zero_grad()
        decoded , feature = model(data)
        #print(decoded.shape, data.shape)
        loss = loss_function(decoded, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (batch_idx+1) % 188 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))


if __name__ == "__main__":

    
    for epoch in range(1, epoch + 1):
        train(epoch, task)

        if (epoch+1)%200 == 0:
            model_name = 'key_feature'
            save_model({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        },
                        os.path.join('savemodel'),model_name)
        '''
        with torch.no_grad():
            sample = torch.randn(192, 16).to(device)
            sample = model.decoder(sample).cpu()
            print(sample.shape)
            save_image(sample.view(64, 3, 32, 32),
                       'results/sample_' + str(epoch) + '.png')
'''