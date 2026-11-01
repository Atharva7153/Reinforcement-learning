import torch
import torch.nn as nn

layer = nn.Linear(2,1)

x = torch.tensor([1.0, 2.0])

result = layer(x)

print("Input : ", x)
print("Output : ", result)


print("Weight : ", layer.weight)
print("Bias : ", layer.bias) 