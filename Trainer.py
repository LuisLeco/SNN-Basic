import torch


class Trainer:
    def __init__(self, net, optimizer, loss_fn, device):
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.loss_hist = []
        self.test_loss_hist = []

    def train_epoch(self, train_loader, encoder, num_steps):
        self.net.train()

        for data, targets in train_loader:
            data = encoder.encode(data).to(self.device)
            targets = targets.to(self.device)

            spk_rec, mem_rec = self.net(data.view(num_steps, data.size(0), -1), num_steps)

            loss_val = torch.zeros((1), dtype=torch.float, device=self.device)
            for step in range(num_steps):
                loss_val += self.loss_fn(mem_rec[step], targets)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            self.loss_hist.append(loss_val.item())

    def evaluate(self, test_loader, encoder, decoder, num_steps):
        self.net.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = encoder.encode(data).to(self.device)
                targets = targets.to(self.device)

                spk_rec, _ = self.net(data.view(num_steps, data.size(0), -1), num_steps)
                decoded = decoder.decode(spk_rec)

                _, predicted = decoded.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Set Accuracy: {accuracy:.2f}%")
        return accuracy
