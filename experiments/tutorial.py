import torch 
import numpy as np

print(f"PyTorch version: {torch.__version__}")

data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)

print(f"Tensor: {tensor}")
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor dtype: {tensor.dtype}")
print(f"Tensor device: {tensor.device}")


np_array = np.array(data)
tensor_from_np = torch.from_numpy(np_array)

print(f"Numpy array: {np_array}")
print(f"Tensor from numpy: {tensor_from_np}")


tensor_from_np.add_(1)
print(f"Numpy array after adding 1: {np_array}")
print(f"Tensor from numpy after adding 1: {tensor_from_np}")


# from another tensor

x_ones = torch.ones_like(tensor)
print(f"Ones Tensor: {x_ones}")

x_rand = torch.rand_like(tensor, dtype=torch.float)
print(f"Random Tensor: {x_rand}")

with random or constant values

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: {rand_tensor}")
print(f"Ones Tensor: {ones_tensor}")
print(f"Zeros Tensor: {zeros_tensor}")

# attributes of a tensor
tensor = torch.rand(3, 4)

print(f"Tensor: {tensor}")
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# tensor to cuda

accelerator = torch.cuda.is_available()
print(f"Accelerator is available: {accelerator}")



if torch.accelerator.is_available():
    print(f"Accelerator is available")
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Tensor is now on {tensor.device}")


# check the GPU type and if CUDA is available

if torch.cuda.is_available():
    print(f"CUDA is available")
    print(f"GPU type: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA is not available")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
