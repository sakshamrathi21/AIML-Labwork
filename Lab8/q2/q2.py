import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

torch.random.manual_seed(42)

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data['Price'] = self.scaler.fit_transform(self.data['Price'].values.reshape(-1,1))

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]['Price'].values
        target = self.data[idx+self.seq_length:idx+self.seq_length+1]['Price'].values
        return torch.tensor(seq).float(), torch.tensor(target).float()
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #TODO: Initialise the RNN layer and a linear fully connected layer to go from hidden_size to output_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        #END TODO

    def forward(self, x, h0):
        #TODO: Implement the forward pass
        
        out = None
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        #END TODO
        return out

def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, print_flag):
    input_size = 1
    hidden_size = 64
    num_layers = 2
    seq_length = 10
    best_loss = np.inf
    val_loss_history = []
    train_loss_history = []
    for epoch in range(num_epochs):
        avg_loss = 0
        model.train()
        for inputs, targets in train_dataloader:
            #TODO: Implement the training loop and update the training loss for each batch
            #reshape inputs as required by the RNN layer
            inputs = inputs.view(-1, seq_length, input_size)
            #initialize hidden state having appropriate dimensions
            h0 = torch.zeros(num_layers, inputs.size(0), hidden_size)
            #forward pass
            outputs = model(inputs, h0)
            loss = criterion(outputs, targets.view(-1, 1))
            #calculate loss and update average loss
            avg_loss += loss.item()
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update weights
            pass
            #END TODO

        avg_loss /= len(train_dataloader)
        train_loss_history.append(avg_loss)

        if (epoch+1) % 10 == 0:
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    #TODO: Implement the validation loop and update the validation loss for each batch
                    inputs = inputs.view(-1, seq_length, input_size)
                    h0 = torch.zeros(num_layers, inputs.size(0), hidden_size)
                    # Forward pass
                    outputs = model(inputs, h0)
                    loss = criterion(outputs, targets.view(-1, 1))
                    val_loss += loss.item()
                    #END TODO

                val_loss /= len(val_dataloader)
                val_loss_history.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    if print_flag:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}, validation loss: {val_loss:.8f} - model saved')
                else:
                    if print_flag:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}, validation loss: {val_loss:.8f}')
        
    return train_loss_history, val_loss_history, best_loss

def test(model, test_dataloader, criterion, print_flag):
    input_size = 1
    hidden_size = 64
    num_layers = 2
    seq_length = 10
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            #TODO: Implement the test loop and update the test loss for each batch
            inputs = inputs.view(-1, seq_length, input_size)
            h0 = torch.zeros(num_layers, inputs.size(0), hidden_size)
            # Forward pass
            outputs = model(inputs, h0)
            loss = criterion(outputs, targets.view(-1, 1))
            test_loss += loss.item()   
            #END TODO

        test_loss /= len(test_dataloader)
        if print_flag:
            print(f'Test Loss: {test_loss:.8f}')
        return test_loss

if __name__ == "__main__":

    data = pd.read_csv("data.csv")

    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

    seq_length = 10
    train_dataset = StockDataset(train_data, seq_length)
    val_dataset = StockDataset(val_data, seq_length)
    test_dataset = StockDataset(test_data, seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    learning_rate = 1e-4
    num_epochs = 200

    model = RNN(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history, val_loss_history, best_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, True)
    model = RNN(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss = test(model, test_dataloader, criterion, True) 
