import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from lstm_ae_toy import LSTMAE
from mnist_loader import MnistLoader

input_size_param = 28
sequence_length = 28
hidden_size = 128
epoch = 5

batch_size = 32

grad_clip = 1
PATH = './MNIST_reconstruct_net.pth'
PATH_2 = './MNIST_classifier_net.pth'
PATH_3 = './MNIST_pixel.pth'

loader = MnistLoader(batch_size, 1)

# MNIST dataset
train_dataset = loader.train_dataset
test_dataset = loader.test_dataset

# Data loader
train_loader = loader.train_loader
test_loader = loader.test_loader


def accuracy_calculator(classifier_lstmae, to_flatten=False):

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size_param)
            # labels = labels.to(device)
            if to_flatten:
                images = torch.flatten(images, start_dim=1)[:, :, None]
            _, outputs = classifier_lstmae(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc


class LSTMAE_CLASSIFIER(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, sequence_size):
        super().__init__()
        self.lstmae = LSTMAE(input_size, hidden_state_size, output_size, sequence_size)
        self.classifier = nn.Linear(hidden_state_size, 10)

    def forward(self, x):
        reconstructed, last_hidden_states = self.lstmae(x)
        classes = self.classifier(last_hidden_states)
        return reconstructed, classes


def train(reconstruct_lstme):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reconstruct_lstme.parameters(), lr=0.001)

    for curr_epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, train_data in enumerate(train_loader, 0):

            inputs, labels = train_data
            inputs = inputs.squeeze()  # [2, 1, 3, 5] -> [2, 3, 5]

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs, _ = reconstruct_lstme(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reconstruct_lstme.parameters(), grad_clip)
            optimizer.step()

            # print statistics
            curr_loss = loss.item()
            running_loss += curr_loss

            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (curr_epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    torch.save(reconstruct_lstme.state_dict(), PATH)
    print('Finished Training, network saved')


def train_classify(classify_lstme, to_flatten=False):
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classify_lstme.parameters(), lr=0.001)
    losses =[]
    iterations = []
    accuracy = []
    for curr_epoch in range(epoch):  # loop over the dataset multiple times
        iterations.append(curr_epoch)
        running_loss = 0.0
        epoch_loss = 0
        j = 0
        for i, train_data in enumerate(train_loader, 0):
            j = i
            inputs, labels = train_data
            inputs = inputs.squeeze()  # [2, 1, 3, 5] -> [2, 3, 5]
            if to_flatten:
                inputs = torch.flatten(inputs, start_dim=1)[:, :, None]


            # zero the parameter gradients
            optimizer.zero_grad()

            reconstructed, classes = classify_lstme(inputs)

            loss = criterion1(reconstructed, inputs)
            loss += criterion2(classes, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classify_lstme.parameters(), grad_clip)
            optimizer.step()

            # print statistics
            curr_loss = loss.item()

            running_loss += curr_loss
            epoch_loss += curr_loss

            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (curr_epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        losses.append(epoch_loss / j)
        accuracy.append(accuracy_calculator(classify_lstme, to_flatten))

    print(losses)
    print(accuracy)
    print(iterations)

    # loss vs. epochs
    plt.plot(iterations, losses, label='row-by-row')
    plt.suptitle('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # accuracy vs. epochs
    plt.plot(iterations, accuracy, label='row-by-row')
    plt.suptitle('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()




    # torch.save(classify_lstme.state_dict(), PATH_2)
    print('Finished Training, network saved')


def task_3_2_1():
    reconstruct_lstme = LSTMAE(input_size_param, hidden_size, input_size_param, sequence_length)

    # train(reconstruct_lstme)

    reconstruct_lstme.load_state_dict(torch.load(PATH))

    dataiter = iter(test_loader)
    input_figures = []
    labels = []
    for i in range(3):
        figure, label = dataiter.next()
        figure = figure.squeeze()
        input_figures.append(figure)
        labels.append(label)
        fig = plt.figure
        print(figure)
        reconstructed_figure, _ = reconstruct_lstme(torch.unsqueeze(input_figures[i], 0))
        plt.imshow(torch.cat((reconstructed_figure.squeeze(), figure), 1).detach().numpy(), cmap='gray')
        plt.title(f'Original vs Reconstructed')
        plt.show()


def task_3_2_2():
    classifier_lstmae = LSTMAE_CLASSIFIER(input_size_param, hidden_size, input_size_param, sequence_length)

    train_classify(classifier_lstmae)

    classifier_lstmae.load_state_dict(torch.load(PATH_2))

    dataiter = iter(test_loader)
    input_figures = []
    input_labels = []
    for i in range(3):
        figure, label = dataiter.next()
        figure = figure.squeeze()
        input_figures.append(figure)
        input_labels.append(label)
        fig = plt.figure
        plt.imshow(figure, cmap='gray')
        plt.title(f'figure number {i}')
        plt.show()

    for i in range(3):
        reconstructed_figure, classes = classifier_lstmae(
            torch.unsqueeze(input_figures[i], 0))  # [28, 28] -> [1, 28, 28]
        fig = plt.figure
        plt.imshow(reconstructed_figure.squeeze().detach().numpy(), cmap='gray')
        plt.title(f'reconstructed figure number {i}, class {classes.argmax()}')
        plt.show()


def task_3_2_3():
    classifier_lstmae = LSTMAE_CLASSIFIER(1, hidden_size, 1, 784)

    train_classify(classifier_lstmae, to_flatten=True)

    classifier_lstmae.load_state_dict(torch.load(PATH_2))

    dataiter = iter(test_loader)
    input_figures = []
    input_labels = []
    for i in range(3):
        figure, label = dataiter.next()
        figure = figure.squeeze()
        input_figures.append(figure)
        input_labels.append(label)
        fig = plt.figure
        plt.imshow(figure, cmap='gray')
        plt.title(f'figure number {i}')
        plt.show()

    for i in range(3):
        reconstructed_figure, classes = classifier_lstmae(
            torch.unsqueeze(input_figures[i], 0))  # [28, 28] -> [1, 28, 28]
        fig = plt.figure
        plt.imshow(reconstructed_figure.squeeze().detach().numpy(), cmap='gray')
        plt.title(f'reconstructed figure number {i}, class {classes.argmax()}')
        plt.show()



# task_3_2_2()
# accuracy_calculator()
# task_3_2_1()
task_3_2_3()
