##########################
# Step 1: Import Libraries
##########################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

##########################
# Step 2: Prepare the data
##########################

data = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Split the data into training and test sets
train_data = np.array(data[:7])
test_data = np.array(data[7:])

###########################################
# Step 3: Create input and output sequences
###########################################

# Define window size
window_size = 3

# Create input and output sequences for training set
train_input_seq = []
train_output_seq = []
for i in range(len(train_data)-window_size):
    train_input_seq.append(train_data[i:i+window_size])
    train_output_seq.append(train_data[i+window_size])

# Convert to numpy arrays
train_input_seq = np.array(train_input_seq)
train_output_seq = np.array(train_output_seq)

# Create input and output sequences for test set
test_input_seq = []
test_output_seq = []
for i in range(len(test_data)-window_size):
    test_input_seq.append(test_data[i:i+window_size])
    test_output_seq.append(test_data[i+window_size])

# Convert to numpy arrays
test_input_seq = np.array(test_input_seq)
test_output_seq = np.array(test_output_seq)

train_input_seq = train_input_seq.reshape(-1, window_size, 1)
test_input_seq = test_input_seq.reshape(-1, window_size, 1)

###############################
# Step 4: Define the LSTM model
###############################

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.fc(lstm_out[-1])
        return predictions

###############################
# Step 5: Instantiate the model and define the loss function and optimizer
###############################

# Define model hyperparameters
input_size = 1
hidden_size = 4
output_size = 1

# Instantiate the model
model = LSTM(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

#########################
# Step 6: Train the model
#########################

# Define number of epochs and batch size
num_epochs = 100
batch_size = 2

# Convert numpy arrays to PyTorch tensors
train_input_seq = torch.from_numpy(train_input_seq).float()
train_output_seq = torch.from_numpy(train_output_seq).float()
test_input_seq = torch.from_numpy(test_input_seq).float()
test_output_seq = torch.from_numpy(test_output_seq).float()

# Train the model
for epoch in range(num_epochs):
    perm_idx = np.random.permutation(len(train_input_seq))
    train_input_seq = train_input_seq[perm_idx]
    train_output_seq = train_output_seq[perm_idx]

    for i in range(0, len(train_input_seq), batch_size):
        inputs = train_input_seq[i:i+batch_size]
        outputs = train_output_seq[i:i+batch_size]

        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds, outputs)

        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print('Training finished!')

############################
# Step 7: Evaluate the model
############################

test_preds = model(test_input_seq)
test_loss = criterion(test_preds.squeeze(), test_output_seq)
print(f'Test Loss: {test_loss.item():.4f}')
