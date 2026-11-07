## Building a Neural Network to predict house pricing

import torch
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(3, 32)
        self.layer2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)

        return x

model = Network()

loss_function = nn.MSELoss()

optimzer = optim.Adam(model.parameters(), lr=0.01)

##Data
X = torch.tensor([
    [500,  1, 10],
    [700,  2,  5],
    [900,  2,  8],
    [1200, 3,  4],
    [1500, 3,  2],
    [1800, 4,  3],
    [2200, 4,  1],
    [2500, 5,  2]
],dtype=torch.float32)

Y = torch.tensor([
    [25],
    [40],
    [45],
    [65],
    [85],
    [100],
    [130],
    [150]
],dtype=torch.float32)

for i in range(10000):

    prediction = model(X)

    loss = loss_function(prediction, Y)

    optimzer.zero_grad()

    loss.backward()

    optimzer.step()

    if i % 100 == 0:
            print(
                f"Epoch : {i}",
                f"Loss : {loss}"
            )

print("After training : ")
test_input = torch.tensor([[2000,4,2]], dtype=torch.float32)

prediction = model(test_input)

print("Prediction : ", prediction.item())
print("Prediction : ", prediction)