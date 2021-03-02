import numpy
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
import torch.nn.functional as F
from torch import optim
from dataset import DeviceDataset
import os
import matplotlib.pyplot as plt
import numpy as np
from complexLayers import ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
#from cplxmodule.nn import CplxBatchNorm1d, RealToCplx, CplxToReal, CplxLinear
from complexFunctions import complex_relu, complex_max_pool2d
import cmath  
import time

torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Net(nn.Module):
  def __init__(self,sequences,features, devices):
      super(Net, self).__init__()
      self.sequences = sequences
      self.features = features
      self.hidden_size = 20
      self.bn0 = BatchNorm2d(self.sequences)
      self.bn1 = BatchNorm2d(self.sequences)
      self.bn2 = BatchNorm2d(self.sequences)
      self.bn3 = BatchNorm2d(self.sequences)
      self.lin1 = ComplexLinear(self.features, self.features)
      self.lstm1 = nn.LSTM(features, self.hidden_size, 1, batch_first = True)
      self.lstm2 = nn.LSTM(features, self.hidden_size, 1, batch_first = True)
      self.lin2 = ComplexLinear(self.hidden_size, self.hidden_size)
      self.output_gate_lin = nn.Linear(self.sequences*self.hidden_size*2,devices)
      self.crelu = complex_relu
      self.relu = nn.ReLU()
      self.soft_out = nn.Softmax(dim = 1)

  def forward(self, x, batch,hidden1=(None,None),hidden2=(None,None)):
      # x : batch_len X self.time X NUM_FEATURES
      #print("0",x)
      h1,c1 = hidden1
      h2,c2 = hidden2
      
      x = torch.view_as_real(x)
      x = self.bn0(x)
      x = torch.view_as_complex(x)
      
      #print(x)
      x = self.lin1(x)
      x = self.crelu(x)
      #print("1",x)
      #x = self.relu(x)
      #print("2",x)
      
      x = torch.view_as_real(x)
      x = self.bn1(x)
      x = torch.view_as_complex(x)
      
      dec1 = x[:,:,::2]
      dec2 = x[:,:,1::2]
      
      x1 = torch.stack((dec1.real,dec2.real), dim = -1).view((batch,self.sequences,self.features))
      x2 = torch.stack((dec1.imag,dec2.imag), dim = -1).view((batch,self.sequences,self.features))
      
      if h1!= None:
        x1,(h1,c1) = self.lstm1(x1,(h1,c1))
        x2,(h2,c2) = self.lstm2(x2,(h2,c2))
      else:
        x1,(h1,c1) = self.lstm1(x1)
        x2,(h2,c2) = self.lstm2(x2)
      
      x = torch.stack((x1,x2), dim = -1)
      
      x = torch.view_as_complex(x).view((batch,self.sequences,self.hidden_size))
      #print("3",x)
      
      x = torch.view_as_real(x)
      x = self.bn2(x)
      x = torch.view_as_complex(x)
      # batch X time x assets*time
      
      x = self.lin2(x)
      x = self.crelu(x)
      
      x = torch.view_as_real(x)
      x = self.bn3(x)
      
      x = x.reshape((batch,self.sequences*self.hidden_size*2))
      x = self.output_gate_lin(x)
      #print("4",x)
      x = self.soft_out(x)
      return x,(h1,c1),(h2,c2)

def data_maker(seq_len, batch_size, num_features):
  days = ['Day 1', 'Day 2', 'Day 3']
  devices = ['Device 1', 'Device 2', 'Device 3']
  raw = []
  for device in devices:
    curr_device = np.array([])
    for day in days:
      curr_dir = day + '/' + device + '/'
      files = os.listdir(curr_dir)
      files = [curr_dir + file for file in files if '.bin' in file]
      for file in files:
        sample = np.fromfile(file, np.complex64)
        #sample = [samp for samp in sample if cmath.isnan(samp) != True]
        if len(sample) % (seq_len*num_features) != 0:
          div_len = int(len(sample)/(seq_len*num_features)) * seq_len*num_features
          sample = sample[:div_len]
        curr_device = np.concatenate((curr_device, sample))
    raw.append(curr_device)
  
  
  targets = [i for i in range(len(devices))]
  data = DeviceDataset(raw, targets, seq_len, batch_size, num_features)
  return data

device = torch.device("cuda:0")

def train(sequence_len = 5000, epochs = 25, lr=1e-3, features = 2):
  net = Net(sequence_len, features, len(data[0][1])).to(device=device)
  optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = 0)
  loss_fn = nn.CrossEntropyLoss()
  
  losses = []
  accuracies = []
  training_accs = []
  for epoch in range(epochs):
    curr_losses = []
    curr_percent = (epoch/epochs) * 100
    print(len(data))
    data_percent = 1/len(data)
    every_ten = int(.1/data_percent)
    print("every_ten: {}".format(every_ten))
    hidden1,hidden2 = (None,None),(None,None)
    start = time.time()
    for i,data_point in enumerate(data):
      samp, targets = data_point
      #samp =  torch.view_as_real(sample).type(torch.FloatTensor).view(batch_size, sequence_len, 2).to(device=device)
      #samp = torch.view_as_real(torch.tensor(samp)).type(torch.FloatTensor).to(device=device)
      samp = samp.to(device=device).view(samp.shape[0], sequence_len, features)
      targets = targets.to(device=device)
      
      out,hidden1,hidden2 = net.forward(samp, len(samp),hidden1,hidden2)
      
      hidden1 = tuple([each.data for each in hidden1])
      hidden2 = tuple([each.data for each in hidden2])
      
      #print(out, targets)
      loss = loss_fn(out,targets)
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
      #print("loss:", loss.item(), loss)
      curr_losses.append(loss.item())
      if np.isnan(loss.item()):
        print('loss is nan index {}'.format(i))
        break
      if i % every_ten == 0:
        run_time = time.time() - start
        print("{}% done on epoch {}".format(100*(i+1)/len(data), epoch))
        per_point_time = run_time/(i + 1)
        time_left = per_point_time * (len(data) - (i + 1))
        print("curr epoch eta: {}".format(time_left))
    sum_losses = sum(curr_losses)
    print('-----------------------------------------------')
    print("epoch {}, loss: {}".format(epoch,sum_losses))
    losses.append(sum_losses)
    
    correct = 0
    for i in range(len(data.test)):
      samp, targets = data.test_set(i)
      samp = samp.to(device=device).view(samp.shape[0], sequence_len, features)
      targets = targets.to(device=device)
      with torch.no_grad():
        out,_,_ = net.forward(samp, samp.shape[0])
        predictions = np.argmax(out.tolist(),axis=1)
        for j,prediction in enumerate(predictions):
          if prediction == targets[j]:
            correct += 1
    accuracy = correct/(len(data.test) * batch_size)
    accuracies.append(accuracy)
    print("validation accuracy:", accuracy)
    correct = 0
    for sample in data:
      samp, targets = sample
      samp = samp.to(device=device).view(samp.shape[0], sequence_len, features)
      targets = targets.to(device=device)
      with torch.no_grad():
        out,_,_ = net.forward(samp, samp.shape[0])
        predictions = np.argmax(out.tolist(),axis=1)
        for j,prediction in enumerate(predictions):
          if prediction == targets[j]:
            correct += 1
    accuracy = correct/(len(data) * batch_size)
    training_accs.append(accuracy)
    print("training accuracy:", accuracy)
    print('-----------------------------------------------')
  
  torch.save(net.state_dict(), 'net.model')
  plt.plot(losses)
  plt.title('Loss Per Epoch lr={}'.format(lr))
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig('plots/loss_{}.png'.format(lr))
  plt.clf()
  
  plt.plot(accuracies)
  plt.title('Validation Accuracy,lr={}'.format(lr))
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.savefig('plots/acc_{}.png'.format(lr))
  plt.clf()
  
  #plt.plot(training_accs)
  #plt.title('Training Accuracy,lr={}'.format(lr))
  #plt.xlabel('Epoch')
  #plt.ylabel('Accuracy')
  #plt.show(block=False)
  return net,accuracies

sequence_len = 1024
batch_size = 10
num_features = 100
data = data_maker(sequence_len, batch_size,num_features)
acc_dict = {}
for lr in [1e-5,5e-4,1e-4]:
  net, accuracies = train(epochs = 25, sequence_len = sequence_len, lr = lr, features = num_features)
  acc_dict[lr] = accuracies

print(acc_dict)
#m = nn.BatchNorm1d(20)
#print(data[0][0])
#samp = torch.view_as_real(data[0][0]).view((5,20))
#print(samp)
