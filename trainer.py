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
from cplxmodule.nn import CplxBatchNorm1d, RealToCplx, CplxToReal, CplxLinear
from complexFunctions import complex_relu, complex_max_pool2d
import cmath  
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Net(nn.Module):
  def __init__(self,sequences,features, devices):
      super(Net, self).__init__()
      self.sequences = sequences
      self.features = features
      self.bn0 = ComplexBatchNorm1d(self.sequences)
      self.bn1 = BatchNorm1d(self.sequences)
      self.bn2 = BatchNorm1d(2*self.sequences*4)
      self.bn3 = BatchNorm1d(self.sequences)
      self.lin1 = ComplexLinear(self.features, self.features)
      self.lstm1 = nn.LSTM(features*2, 4, 1, batch_first = True)
      self.lstm2 = nn.LSTM(features*2, 4, 1, batch_first = True)
      self.lin2 = nn.Linear(self.sequences*4*2, devices)
      self.crelu = complex_relu
      self.relu = nn.ReLU()
      self.soft_out = nn.Softmax(dim = 1)

  def forward(self, x, batch):
      # x : batch_len X self.time X NUM_FEATURES
      #print("0",x)
      #x = self.bn0(x)
      x = self.lin1(x)
      #print("1",x)
      #x = self.relu(x)
      x = self.crelu(x)
      #print("2",x)
      #x = self.bn1(x)
      x = torch.view_as_real(x).view((batch,sequence_len,self.features*2))
      x1,(_,_) = self.lstm1(x)
      x2,(_,_) = self.lstm2(x)
      x = torch.stack((x1,x2)).reshape((2*batch,self.sequences,4))
      #print("3",x)
      x = torch.reshape(x, (batch, 2*self.sequences*4))
      x = self.bn2(x)
      # batch X time x assets*time
      x = self.lin2(x)
      #print("4",x)
      x = self.soft_out(x)
      return x

def data_maker(seq_len, batch_size):
  #days = ['Day 1', 'Day 2', 'Day 3']
  #devices = ['Device 1', 'Device 2', 'Device 3']
  days = ['Day 1']
  devices = ['Device 1', 'Device 2']
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
        curr_device = np.concatenate((curr_device, sample))
    raw.append(curr_device)
  
  
  targets = [i for i in range(len(devices))]
  data = DeviceDataset(raw, targets, seq_len, batch_size)
  return data

device = torch.device("cuda:0")

def train(sequence_len = 5000, epochs = 25, lr=1e-3, features = 2):
  net = Net(sequence_len, features, 2).to(device=device)
  optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = 0)
  loss_fn = nn.CrossEntropyLoss()
  
  losses = []
  accuracies = []
  for epoch in range(epochs):
    curr_losses = []
    curr_percent = (epoch/epochs) * 100
    print(len(data))
    data_percent = 1/len(data)
    every_ten = int(.1/data_percent)
    print("every_ten: {}".format(every_ten))
    start = time.time()
    for i,data_point in enumerate(data):
      samp, targets = data_point
      #samp =  torch.view_as_real(sample).type(torch.FloatTensor).view(batch_size, sequence_len, 2).to(device=device)
      #samp = torch.view_as_real(torch.tensor(samp)).type(torch.FloatTensor).to(device=device)
      samp = samp.to(device=device).view(batch_size, sequence_len, features)
      targets = targets.to(device=device)
      
      out = net.forward(samp, len(samp))
      #print(out, targets)
      loss = loss_fn(out,targets)
      optimizer.zero_grad()
      loss.backward()
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
        out = net.forward(samp, len(samp))
        predictions = np.argmax(out.tolist(),axis=1)
        for j,prediction in enumerate(predictions):
          if prediction == targets[j]:
            correct += 1
    accuracy = correct/(len(data.test) * batch_size)
    print("accuracy:", accuracy)
    accuracies.append(accuracy)
    print('-----------------------------------------------')
  
  torch.save(net.state_dict(), 'net.model')
  plt.plot(losses)
  plt.show()
  plt.plot(accuracies)
  plt.show()
  return net

sequence_len = 5000
batch_size = 40
data = data_maker(sequence_len, batch_size)
net = train(epochs = 10, sequence_len = sequence_len, lr = 1e-4, features = 1)


#m = nn.BatchNorm1d(20)
#print(data[0][0])
#samp = torch.view_as_real(data[0][0]).view((5,20))
#print(samp)
