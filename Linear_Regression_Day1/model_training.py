# Model Training for Linear Regression on the Car Price Dataset using PyTorch

from load_dataset import CarPriceDataset
import torch
import torch.nn as nn
import torch.optim as optim
from simple_scaler import SimpleStandardScaler
import numpy as np

dataset = CarPriceDataset(convert_to_tensor=True)
dataset.load_data()

print(dataset.get_data()[:5])

test_data = dataset.get_data()
feature_data = test_data[:, :-1]
target_data = test_data[:, -1]

print("feature_data.shape", feature_data.shape)
print("target_data.shape", target_data.shape)

# Normalize the data using simple scaler for M1 compatibility
print("Normalizing features...")
feature_scaler = SimpleStandardScaler()
feature_data_normalized = feature_scaler.fit_transform(feature_data)

print("Normalizing targets...")
target_scaler = SimpleStandardScaler()
target_data_normalized = target_scaler.fit_transform(target_data.unsqueeze(1))

print(f"Original target range: {target_data.min():.2f} to {target_data.max():.2f}")
print(f"Normalized target range: {target_data_normalized.min():.2f} to {target_data_normalized.max():.2f}")

# Write the linear regression model using pytorch

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        # Simplified architecture - start simple
        self.linear = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout to prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.linear(x)
        

# Keep 5% test data and 95% training data
split_index = int(feature_data_normalized.shape[0] * 0.05)

test_features = feature_data_normalized[:split_index]
test_target = target_data_normalized[:split_index]
print("test_features.shape", test_features.shape)
print("test_target.shape", test_target.shape)

training_features = feature_data_normalized[split_index:]
training_target = target_data_normalized[split_index:]
print("training_features.shape", training_features.shape)
print("training_target.shape", training_target.shape)

# Write a training loop

def train_model(model, features, target, epochs=1000, learning_rate=0.001):
    # Use a lower learning rate
    # Use weight decay to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {epochs} epochs with learning rate {learning_rate}...")
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 5000 == 0:
            avg_loss = np.mean(losses[-5000:])  # Average of last 1000 losses
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}')

    return model, loss, losses

# Write a testing loop

def test_model(model, features, target):
    # set model to evaluation mode
    model.eval()
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        outputs = model(features)
        loss = criterion(outputs, target)
        
        # Calculate some additional metrics
        mae = torch.mean(torch.abs(outputs - target))
        mape = torch.mean(torch.abs((target - outputs) / (target + 1e-8))) * 100
        
        return loss, mae, mape

# Get Accuracy of the model

def get_accuracy(model, features, target):
    # For regression, we use loss instead of accuracy
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = model(features)
        loss = criterion(outputs, target)
    return loss

# Create and train the model
input_size = feature_data_normalized.shape[1]
output_size = 1

print(f"Creating model with input_size: {input_size}, output_size: {output_size}")

model = LinearRegressionModel(input_size, output_size)
print(f"Model created: {model}")

# Train the model with lower learning rate and fewer epochs
trained_model, final_loss, loss_history = train_model(
    model, training_features, training_target, 
    epochs=50000, learning_rate=0.001
)

print(f"Training completed! Final loss: {final_loss.item():.6f}")

# Test the model
test_loss, test_mae, test_mape = test_model(trained_model, test_features, test_target)

print(f"Test Results:")
print(f"  MSE Loss: {test_loss.item():.6f}")
print(f"  MAE: {test_mae.item():.6f}")
print(f"  MAPE: {test_mape.item():.2f}%")

# Make some predictions and convert back to original scale
trained_model.eval()
with torch.no_grad():
    sample_predictions_normalized = trained_model(test_features[:5])
    
    # Convert predictions back to original scale
    sample_predictions = target_scaler.inverse_transform(sample_predictions_normalized.numpy())
    sample_actuals = target_scaler.inverse_transform(test_target[:5].numpy())
    
    print(f"\nSample predictions vs actual (original scale):")
    for i in range(5):
        print(f"  Predicted: {sample_predictions[i][0]:.2f}, Actual: {sample_actuals[i][0]:.2f}")

        

    
