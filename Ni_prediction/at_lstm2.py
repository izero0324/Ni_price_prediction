import torch
import os
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from optimizer import Ranger
import datetime
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
#from autoencoder import auto_encoder_FFN1
from dataload2 import price_list, target_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def save_model(state,save_model_path,modelname):
    filename = os.path.join(save_model_path,modelname+'_'+str(state['epoch']+1)+'.pth')
    torch.save(state,filename)

def load_model(Net, optimizer, model_file):
    assert os.path.exists(model_file),'There is no model file from'+model_file
    checkpoint = torch.load(model_file, map_location='cuda:0')
    Net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']+1
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return Net, optimizer, start_epoch





class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, time_step):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention = nn.Linear(2*hidden_size+time_step-1, 1)
        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size)
        self.T = time_step

    def forward(self, input_):
        input_ = input_.unsqueeze(0)
        input_weight = Variable(input_.data.new(input_.size(0), self.T, self.input_size).zero_())
        input_encode = Variable(input_.data.new(input_.size(0), self.T, self.hidden_size).zero_())

        hidden = self.init_hidden(input_)
        cell = self.init_hidden(input_)

        for t in range(self.T-1):

            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1,0,2),
                           cell.repeat(self.input_size, 1, 1).permute(1,0,2),
                           input_.permute(0,2,1)), dim = 2)
            #print('x', x.shape)

            x = self.attention(x.view(-1, self.hidden_size*2 + self.T-1))
            #print('x', x.shape)
            attention_w = F.softmax(x.view(-1, self.input_size))
            #print(attention_w.shape, x.shape)
            weight_input = torch.mul(attention_w, input_[:,t,:])
            self.encoder.flatten_parameters()
            _, lstm_states = self.encoder(weight_input.unsqueeze(0), (hidden,cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]

            input_weight[:,t,:] = weight_input
            input_encode[:,t,:] = hidden
        return input_weight, input_encode

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_())

class LSTM_price(nn.Module):
    """docstring for LSTM_price"""
    def __init__(self, hidden_size = 45 , predict_size = 45):
        super(LSTM_price, self).__init__()
        self.price_encoder = lstm_encoder(45, 45, 32).to(device)
        
        self.price_input = 45
        self.key_input = 0
        self.T = 32
        self.hidden_size = hidden_size
        self.predict_size = predict_size
        self.atten_layer = nn.Sequential(nn.Linear(2*hidden_size + self.price_input + self.key_input, self.price_input + self.key_input),
                                         nn.Tanh(), nn.Linear(self.price_input + self.key_input, 1))
        self.lstm = nn.LSTM(input_size = self.price_input + self.key_input, hidden_size = hidden_size)
        self.fc = nn.Linear(2*hidden_size, predict_size)
        self.convert_ch = nn.Conv1d(31,30, kernel_size = 1)

    def forward(self, price_input):
        #print(price_input)
        _, price_vec = self.price_encoder(price_input)
        #print(price_vec)
    
        #print(price_vec.shape, key_vec.shape)
        in_vec = price_vec
        hidden = self.init_hidden(in_vec)
        cell = self.init_hidden(in_vec)

        

        x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1,0,2),
                      cell.repeat(self.T, 1, 1).permute(1,0,2), in_vec), dim = 2)
        x = F.softmax(self.atten_layer(x.view(-1, 2*self.hidden_size + self.price_input + self.key_input)).view(-1, self.T))
        context = torch.bmm(x.unsqueeze(1), in_vec)[:, 0, :]
        self.lstm.flatten_parameters()
        _, lstm_output = self.lstm(context.unsqueeze(0), (hidden, cell))
        hidden = lstm_output[0]
        cell = lstm_output[1]

        y_pred = self.fc(torch.cat((hidden[0], context), dim = 1))

        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_())

        
        



def train(price_tensor, target_tensor, decoder, optimizer, criterion, num_epochs):
    
    price_tensor = torch.from_numpy(price_tensor).float().to(device)
    target_tensor =  torch.from_numpy(target_tensor).float().to(device)
    optimizer.zero_grad()

    input_length = price_tensor.size(0)
    target_length = target_tensor.size(0)
    best_loss = 10e100

    for epoch in range(num_epochs):
        total_loss = 0
        for ei in range(input_length):

            decoder_output = decoder(price_tensor[ei])         
            #print(decoder_output.shape, target_tensor[ei].shape)
            loss = criterion(decoder_output, target_tensor[ei])
            if ei == input_length:
               loss = loss*(input_length-1)/2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(decoder_output, target_tensor[ei])
        average_loss = total_loss/target_length
        print('epoch:', epoch, 'loss:', average_loss)

        if epoch == 10:
            save_model({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        },
                        os.path.join('savemodel'),model_name)
            best_loss = average_loss
            pass
        if (epoch+1)%200 == 0:
            
            model_name = 'price_predict2'
            save_model({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        },
                        os.path.join('savemodel'),model_name)

        if average_loss < best_loss:
            best_loss = average_loss
            model_name = 'bst_mod'
            save_model({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        },os.path.join('savemodel'),model_name)
            pass
    

    

    return average_loss

if __name__ == "__main__":

    
    model = LSTM_price().to(device)
    loss_fn = nn.MSELoss()
    epochs = 1000
    price = price_list()
    target = target_list()
    optimizer = Ranger(model.parameters())
    train_loss = train(price, target, model, optimizer, loss_fn, epochs)