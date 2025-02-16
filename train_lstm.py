import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils.utils import training_details, create_dir_ml

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("lens.log"),
                              logging.StreamHandler()])

data = pd.read_excel('results/dbspl_to_rpm.xlsx')
x_data = np.array(data['weighted_time_freq_dbspl'])
y_data = np.array(data['x'])

# Convert arrays to tensors
x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)
y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

x_mean, x_std = x_tensor.mean(), x_tensor.std()
y_mean, y_std = y_tensor.mean(), y_tensor.std()

x_tensor = (x_tensor - x_mean) / x_std
y_tensor = (y_tensor - y_mean) / y_std

# Split data
split_index = int(0.8 * len(x_tensor))
x_train = x_tensor[:split_index]
y_train = y_tensor[:split_index]
x_val = x_tensor[split_index:]
y_val = y_tensor[split_index:]

# Define the model
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = nn.Linear(128, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = LSTM()

# Specify loss and optimizer
criterion = nn.MSELoss()
lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    train_outputs = model(x_train)
    train_loss = criterion(train_outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        # Compute validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluate the model

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted = model(x_tensor).detach().numpy()
    predicted_original = predicted * float(y_std) + float(y_mean)

# Compare predictions with actual data
logging.info(f"Predicted values: {predicted_original}")
logging.info(f"Actual values: {y_data}")

plt.figure(figsize=(10, 6))
plt.plot(y_data, color='blue', label='Actual')
plt.plot(predicted_original, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

dir = create_dir_ml()
training_details(logging, model, optimizer, criterion, num_epochs, lr, dir, plt) #Logs the training details and creates a json file

plt.show()