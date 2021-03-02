import random 
from torch.utils.data import Dataset
import torch
#from sklearn.preprocessing import MinMaxScaler
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DeviceDataset(Dataset):
    def __init__(self, inputs, target, seq_len, batch_size, num_features = 100):
        self.samples = []
        #scaler = MinMaxScaler()
        for index,device in enumerate(inputs):
            sequences = []
            for i in range(0, len(device), seq_len*num_features):
              select = device[i:i+(seq_len*num_features)]
            #  if len(select) == seq_len*num_features*decimation:
            #    for j in range(decimation):
            #      curr = [select[i] for i in range(j, len(select),decimation)]
            #      sequences.append((curr,target[index]))
            
              sequences.append((select,target[index]))
                
              #scaler.fit(select)
              #normalized = scaler.transform(select)
              #sequences.append((normalized,target[index]))
              
            
            if len(sequences[-1][0]) != seq_len*num_features:
              sequences=sequences[:-1]
            self.samples.extend(sequences)
        random.shuffle(self.samples)
        self.samples = [self.samples[i:i+batch_size] for i in range(0,len(self.samples), batch_size)]
        test_set_len = int(0.2 * len(self.samples))
        self.test = self.samples[-test_set_len:]
        self.samples = self.samples[:-test_set_len]
        
        
    def __getitem__(self, idx):
        items = self.samples[idx]
        samples = []
        targets = []
        for item in items:
          samples.append(item[0])
          targets.append(item[1])
        return (torch.tensor(samples, dtype=torch.cfloat),torch.tensor(targets))
      
    def test_set(self,idx):
        items = self.test[idx]
        samples = []
        targets = []
        for item in items:
          samples.append(item[0])
          targets.append(item[1])
        return (torch.tensor(samples, dtype=torch.cfloat),torch.tensor(targets))
      
    def __len__(self):
        return len(self.samples)
