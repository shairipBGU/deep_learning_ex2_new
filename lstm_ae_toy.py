import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from synthetic_data_generator import synthetic_data_gen

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PATH = './lstm_ae_net.pth'
batch_size = 30
epoch = 50
grad_clip = 1
learning_rate = 0.001
hidden_state_size_param = 40
input_size_param = 1


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, last_enc_hidden_state=False):
        super().__init__()
        self.last_enc_hidden_state = last_enc_hidden_state
        self.lstm_enc = nn.LSTM(input_size, hidden_state_size, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_state_size, hidden_state_size, batch_first=True)
        self.fc = nn.Linear(hidden_state_size, output_size)

    def forward(self, x):
        output_enc, (_, _) = self.lstm_enc(x)
        _, sequence_size, k = output_enc.size()
        to_repeat = output_enc[:, -1, :]
        to_repeat = to_repeat[:, None, :]
        z = to_repeat.repeat(1, sequence_size, 1)
        last_hidden_states = z
        output_dec, (_, _) = self.lstm_dec(z)
        if not self.last_enc_hidden_state:
            last_hidden_states = output_dec[:, -1, :]

        output = self.fc(output_dec)

        return output, last_hidden_states


def train(lstmae):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstmae.parameters(), lr=0.001)

    train_data = synthetic_data_gen()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    losses = []
    iterations = []
    for curr_epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, train_data in enumerate(train_loader, 0):

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs, _ = lstmae(train_data)

            loss = criterion(outputs, train_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstmae.parameters(), grad_clip)
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

    torch.save(lstmae.state_dict(), PATH)
    print('Finished Training, network saved')


def task_3_2():
    lstmae = LSTMAE(input_size_param, hidden_state_size_param, input_size_param)
    train(lstmae)

    test_data = synthetic_data_gen()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    dataiter = iter(test_loader)
    signal = dataiter.next()

    axis = [i for i in range(50)]

    original = signal.flatten().detach().numpy()
    plt.plot(axis, original, label='original signal')
    plt.suptitle(f'batch size : {batch_size} epoch: {epoch} gradient clipping: {grad_clip}')
    plt.xlabel('Time')
    plt.ylabel('Value')

    lstmae.load_state_dict(torch.load(PATH))

    result, _ = lstmae(signal)
    reconstructed_signal = result.flatten().detach().numpy()
    plt.plot(axis, reconstructed_signal, label='estimated signal')
    plt.legend()
    plt.show()

# task_3_2()
