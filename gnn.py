import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define a custom dataset
class MolDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# Define a simple GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.gnn_layer = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.gnn_layer(x)
        x = self.fc(x[:, -1, :])
        return x

# Create synthetic dummy data for demonstration
sequence_length = 10  # Define the sequence length
data = np.random.rand(100, sequence_length, 256).astype(np.float32)
target_values = np.random.rand(100).astype(np.float32)

# Create dataset and dataloader
dataset = MolDataset(data, target_values)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize and train the GNN model
input_dim = 256  # Change according to your feature representation
hidden_dim = 128
output_dim = 1
model = GNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(data)

