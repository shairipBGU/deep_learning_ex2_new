import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from lstm_ae_toy import LSTMAE

sequence_param = 19
grad_clip = 1
learning_rate = 0.001
hidden_state_size_param = 512
input_size_param = 53
epoch = 200
batch_size = 10
PATH = './snp500_net.pth'

stocks = pd.read_csv('./SP 500 Stock Prices 2014-2017.csv')
df_stocks_high = stocks.pivot(index='date', columns='symbol', values='high').dropna(axis=1, how='any')
df = pd.DataFrame({'values': np.random.randint(0, 1000, 1007)},
                  index=pd.date_range(start='2014-01-02', end='2017-12-29', periods=1007))

transposed_data = np.asarray(df_stocks_high.values).T
reshaped_data = np.reshape(transposed_data, (477, sequence_param, input_size_param))
to_normalized_data = torch.FloatTensor(reshaped_data)
data = torch.nn.functional.normalize(to_normalized_data)

# train set 2/3, test set 1/3
train_set, test_set = torch.utils.data.random_split(data, [318, 159])


loader = torch.utils.data.DataLoader(dataset=train_set,
                                     batch_size=batch_size,
                                     shuffle=True)
data_iter = iter(loader)
signal = data_iter.next()


def train(lstmae):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstmae.parameters(), lr=learning_rate)

    losses = []
    iterations = []
    for curr_epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, train_data in enumerate(loader, 0):
            # print("data shape", data.shape)
            # print("data", data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # tensor_first_3 = torch.tensor(first_3)
            outputs, _ = lstmae(train_data)
            # print("output", outputs.shape)
            # print("first_3", tensor_first_3.shape)
            loss = criterion(outputs, train_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstmae.parameters(), grad_clip)
            optimizer.step()

            # print statistics
            curr_loss = loss.item()
            running_loss += curr_loss
            losses.append(curr_loss)
            iterations.append((curr_epoch + i))
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (curr_epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    torch.save(lstmae.state_dict(), PATH)
    print('Finished Training, network saved')


def task_3_3_2():
    lstmae = LSTMAE(input_size_param, hidden_state_size_param, input_size_param, sequence_param)
    train(lstmae)

    test_data = data
    test_data = test_set
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    dataiter = iter(test_loader)
    stock = dataiter.next()

    fig, ax1 = plt.subplots()
    original = stock.flatten().detach().numpy()
    plt.plot(df.index, original, label='original stock')
    plt.suptitle('Original Vs. Reconstructed Stock')
    plt.ylabel('Price')

    lstmae.load_state_dict(torch.load(PATH))

    result, _ = lstmae(stock)
    reconstructed_signal = result.flatten().detach().numpy()
    plt.plot(df.index, reconstructed_signal, label='estimated stock')
    monthyearFmt = mdates.DateFormatter('%Y %B')
    ax1.xaxis.set_major_formatter(monthyearFmt)
    _ = plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def PrintGoogleAndAmazonMaximums():
    google = df_stocks_high['GOOGL'].values
    amzn = df_stocks_high['AMZN'].values
    date = df_stocks_high.index
    axis = [i for i in range(1007)]

    fig, ax1 = plt.subplots()
    plt.plot(df.index, google, label='Google')
    plt.plot(df.index, amzn, label='Amazon')
    monthyearFmt = mdates.DateFormatter('%Y %B')
    ax1.xaxis.set_major_formatter(monthyearFmt)
    _ = plt.xticks(rotation=90)
    plt.title('Google and Amazon daily maximums')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# PrintGoogleAndAmazonMaximums()
task_3_3_2()
