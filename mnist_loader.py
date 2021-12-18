import torch
import torchvision
import torchvision.transforms as transforms


class MnistLoader:
    def __init__(self, train_batch_size, test_batch_size):
        # MNIST dataset
        self.train_dataset = torchvision.datasets.MNIST(root='./data',
                                                        train=True,
                                                        transform=transforms.ToTensor(),
                                                        download=True)

        self.test_dataset = torchvision.datasets.MNIST(root='./data',
                                                       train=False,
                                                       transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=train_batch_size,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=test_batch_size,
                                                       shuffle=False)
