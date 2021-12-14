import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from lstm_ae_toy import LSTMAE

input_size = 28
sequence_length = 28
hidden_size = 128
epoch = 20
batch_size = 32
PATH = './MNIST_reconstruct_net.pth'

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)


def train(reconstruct_lstme):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reconstruct_lstme.parameters(), lr=0.001)

    losses = []
    iterations = []
    for curr_epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, train_data in enumerate(train_loader, 0):

            inputs, labels = train_data
            inputs = inputs.squeeze()  # [2, 1, 3, 5] -> [2, 3, 5]

            # print("inputs shape", inputs.shape)
            # print("data", data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # tensor_first_3 = torch.tensor(first_3)
            outputs = reconstruct_lstme(inputs)
            # print("output", outputs.shape)
            # print("first_3", tensor_first_3.shape)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            curr_loss = loss.item()
            running_loss += curr_loss
            losses.append(curr_loss)
            iterations.append((curr_epoch + i))
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (curr_epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    torch.save(reconstruct_lstme.state_dict(), PATH)
    print('Finished Training, network saved')


def task_3_2_1():
    reconstruct_lstme = LSTMAE(input_size, hidden_size, input_size, sequence_length)

    # train(reconstruct_lstme)

    reconstruct_lstme.load_state_dict(torch.load(PATH))

    dataiter = iter(test_loader)
    input_figures = []
    labels = []
    for i in range(3):
        figure, labels = dataiter.next()
        figure = figure.squeeze()
        input_figures.append(figure)
        fig = plt.figure
        plt.imshow(figure, cmap='gray')
        plt.title(f'figure number {i}')
        plt.show()

    for i in range(3):
        reconstructed_figure = reconstruct_lstme(torch.unsqueeze(input_figures[i], 0))  # [28, 28] -> [1, 28, 28]
        fig = plt.figure
        plt.imshow(reconstructed_figure.squeeze().detach().numpy(), cmap='gray')
        plt.title(f'reconstructed figure number {i}')
        plt.show()

task_3_2_1()