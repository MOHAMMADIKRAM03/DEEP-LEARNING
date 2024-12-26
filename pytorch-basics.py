# From Numpy to PyTorch

import numpy as np    # import numpy
import torch          # import torch

array = [[1, 2, 3], [4, 5, 6]]  # initial data

np_array = np.array(array)
print("Numpy Array Type: {}".format(type(np_array)))
print("Numpy Array Shape: {}".format(np_array.shape))
print("Numpy Array:")
print(np_array)

tensor = torch.Tensor(array)
print("PyTorch Array Type: {}".format(tensor.type))
print("PyTorch Array Shape: {}".format(tensor.shape))
print("PyTorch Array:")
print(tensor)

"""# Initialize Matrix (Tensor)"""

np_ones = np.ones((2,3))
print(np_ones)

torch_ones = torch.ones((2,3))
print(torch_ones)

print(torch.arange(10))   # equivalent to np.arange
print(torch.rand(3, 4))   # euivalent to np.random.rand

"""# Conversion between Numpy and PyTorch"""

np_array = np.random.rand(2, 3)
print(np_array)
print(type(np_array))

tensor_from_np_array = torch.from_numpy(np_array)   # From numpy to pytorch
print(tensor_from_np_array)
print(tensor_from_np_array.type)

np_array_from_tensor = tensor_from_np_array.numpy()
print(np_array_from_tensor)
print(type(np_array_from_tensor))
print(np.allclose(np_array, np_array_from_tensor))

"""# Basic Operations on Tensors"""

a = torch.rand(3, 4)
b = torch.ones(3, 4)

print(a)
print(b)

print(a.reshape(2, 6))  # Reshape
print(a + b)            # Addition
print(a - b)            # Subtraction
print(a * b)            # Element-wise multiplication
print(b / a)            # Element-wise division
print(a @ b.T)          # Matrix multiplication
print((b.T @ a).mean()) # Mean
print((b.T @ a).std())  # Standard deviation

print(a.mean())
print(a.mean().item())          # Trick: convert 0-d tensor into scalar
print(type(a.mean().item()))

"""# Move to GPU for Accelerated Computation"""

# Commented out IPython magic to ensure Python compatibility.
# %%timeit
# a = torch.rand(1000, 100000)
# b = torch.rand(100000, 1000)
# c = a @ b

# Commented out IPython magic to ensure Python compatibility.
# %%timeit
# a = torch.rand(1000, 100000).cuda()
# b = torch.rand(100000, 1000).cuda()
# c = a @ b

cpu_tensor = torch.rand(5)
#gpu_tensor = cpu_tensor.cuda()
print(cpu_tensor)
#print(gpu_tensor)

# Let's check what GPUs we got
#torch.cuda.get_device_name(0)

# And CPU info
#!cat /proc/cpuinfo

"""# Automatic Differentiation"""

x = torch.tensor([1., 2., 3, 4], requires_grad=True)
y = torch.tensor([4., 5., 6, 7], requires_grad=True)
z = 5 * x + 2 * y

m = z.sum()
print(z)

m.backward()
print(x.grad)
print(y.grad)



m = torch.sum(x * (y ** 2) - 4 * x * y)
print(x)
print(y)
print(m)      # So you can write arbitrarily complicated expression!

x.grad.zero_()   # Gradients will be accumulated, so we need to clear out first.
y.grad.zero_()

m.backward()
print(x.grad)
print(y.grad)