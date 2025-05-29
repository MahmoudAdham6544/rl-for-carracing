import torch
from agent.networks import CNN


class BCAgent:

    def __init__(self):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = CNN().to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        X_batch = torch.tensor(X_batch, dtype=torch.float32)
        y_batch = torch.tensor(y_batch, dtype=torch.long)

        # move to device
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # TODO: forward + backward + optimize
        self.net.train()
        self.optimizer.zero_grad()
        outputs = self.net(X_batch)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        self.net.eval()

        X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)
        # add batch dim if not present
        if len(X.shape) == 3:
            X = X.unsqueeze(0)

        with torch.no_grad():
            net_outputs = self.net(X)
            outputs = torch.argmax(net_outputs, dim=1)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
