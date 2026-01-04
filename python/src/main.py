import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32)

y = torch.tensor([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
], dtype=torch.float32)


class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)  # 2 inputs -> 4 hidden neurons
        self.fc2 = nn.Linear(4, 2)  # 4 hidden -> 2 output
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


model = XORNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    predictions = model(X)
    print("\nPredictions:")
    for inp, out in zip(X, predictions):
        print(f"{inp.tolist()} -> {[round(item, 3) for item in out.tolist()]}")

torch.save(model.state_dict(), "xor_model_weights.pth")
print("Weights and biases saved to xor_model_weights.pth")
