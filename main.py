import torch
# import concepts of Pytorch 


# create a tensor


x =  torch.tensor([1,2,3])

# print(x)

# create a tensor of zeros and onces

zeros_tensor =  torch.zeros(2,4)     # 2,4 are rows and columns respectively
# print(zeros_tensor)

ones_tensors = torch.ones(2,4)

# print(ones_tensors)


# get shape of tensors

# print(ones_tensors.shape,ones_tensors.dtype)

# Gradient 

t = torch.tensor(2.0,requires_grad=True)

# print(t)

y = t ** 2 + 3* t + 1  # t^2 +3t + 1    (quadratic equation)
# print(y)

y.backward()  #  computes dy/dt    differentiation of y wrt  t 

print(t.grad)  # t*2+3 = 7