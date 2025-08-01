# Model Training for Linear Regression on the Car Price Dataset using PyTorch

from load_dataset import CarPriceDataset
import torch
import torch.nn as nn

dataset = CarPriceDataset(convert_to_tensor=True)
dataset.load_data()

print(dataset.get_data()[:5])

test_data = dataset.get_data()
feature_data = test_data[:, :-1]
target_data = test_data[:, -1]

print("feature_data.shape", feature_data.shape)
print("target_data.shape", target_data.shape)

# Write the linear regression model using pytorch

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
        

# Keep 5% test data and 95% training data
split_index = int(feature_data.shape[0] * 0.05)

test_features = feature_data[:split_index]
test_target = target_data[:split_index]
print("test_features.shape", test_features.shape)
print("test_target.shape", test_target.shape)

training_features = feature_data[split_index:]
training_target = target_data[split_index:]
print("training_features.shape", training_features.shape)
print("training_target.shape", training_target.shape)

# Reshape target data to match model output
training_target = training_target.unsqueeze(1)  # Add dimension to match model output
test_target = test_target.unsqueeze(1)

print("training_target.shape after reshape:", training_target.shape)
print("test_target.shape after reshape:", test_target.shape)

# Write a training loop

def train_model(model, features, target, epochs=1000, learning_rate=0.01):
    # Write a loss function with ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model, loss

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
        mape = torch.mean(torch.abs((target - outputs) / target)) * 100
        
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
input_size = feature_data.shape[1]
output_size = 1

print(f"Creating model with input_size: {input_size}, output_size: {output_size}")

model = LinearRegressionModel(input_size, output_size)
print(f"Model created: {model}")

# Train the model
trained_model, final_loss = train_model(model, training_features, training_target, epochs=500000, learning_rate=0.01)

print(f"Training completed! Final loss: {final_loss.item():.4f}")

# Test the model
test_loss, test_mae, test_mape = test_model(trained_model, test_features, test_target)

print(f"Test Results:")
print(f"  MSE Loss: {test_loss.item():.4f}")
print(f"  MAE: {test_mae.item():.4f}")
print(f"  MAPE: {test_mape.item():.2f}%")

# Make some predictions
trained_model.eval()
with torch.no_grad():
    sample_predictions = trained_model(test_features[:5])
    print(f"\nSample predictions vs actual:")
    for i in range(5):
        print(f"  Predicted: {sample_predictions[i][0]:.2f}, Actual: {test_target[i][0]:.2f}")
        

    
