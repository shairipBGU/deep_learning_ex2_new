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
raw_data = torch.FloatTensor(reshaped_data)
# Normalization of the data
min_val = raw_data.min(1, keepdim=True)[0]
max_val = raw_data.max(1, keepdim=True)[0]
data = (raw_data - min_val)/max_val

# train set 80%, test set 20%
# train_set, test_set = torch.utils.data.random_split(data, [382, 95])
train_set = data[:-95, :, :]
test_set = data[382:, :, :]


loader = torch.utils.data.DataLoader(dataset=train_set,
                                     batch_size=batch_size,
                                     shuffle=True)
data_iter = iter(loader)
signal = data_iter.next()


class LSTMAE_PREDICTOR(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, sequence_size):
        super().__init__()
        self.lstmae = LSTMAE(input_size, hidden_state_size, output_size, sequence_size, True)
        self.lstm_dec_predictor = nn.LSTM(hidden_state_size, hidden_state_size, batch_first=True)
        self.fc_predictor = nn.Linear(hidden_state_size, output_size)

    def forward(self, x):
        reconstructed, z = self.lstmae(x)
        output_dec, (_, _) = self.lstm_dec_predictor(z)
        prediction = self.fc_predictor(output_dec)
        return reconstructed, prediction


def train_predict(lstmae_predictor):
    reconstruct_criterion = nn.MSELoss()
    prediction_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstmae_predictor.parameters(), lr=learning_rate)

    reconstruction_losses = []
    prediction_losses = []
    reconstruction_epoch_loss = 0.0
    prediction_epoch_loss = 0.0
    iterations = []
    for curr_epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        j = 0
        for i, train_data in enumerate(loader, 0):
            j = i
            # print("data shape", data.shape)
            # print("data", data)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # tensor_first_3 = torch.tensor(first_3)
            x_t = train_data[:, :-1, :]
            y_t = train_data[:, 1:, :]

            reconstructed, prediction = lstmae_predictor(x_t)
            # print("output", outputs.shape)
            # print("first_3", tensor_first_3.shape)
            reconstruction_loss = reconstruct_criterion(reconstructed, x_t)
            prediction_loss = prediction_criterion(prediction, y_t)
            loss = prediction_loss + reconstruction_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstmae_predictor.parameters(), grad_clip)
            optimizer.step()

            # print statistics
            curr_loss = loss.item()
            running_loss += curr_loss

            reconstruction_epoch_loss += reconstruction_loss.item()
            prediction_epoch_loss += prediction_loss.item()


            iterations.append((curr_epoch + i))
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (curr_epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        reconstruction_losses.append(reconstruction_epoch_loss / j)
        prediction_losses.append(prediction_epoch_loss / j)
    torch.save(lstmae_predictor.state_dict(), PATH)
    print('Finished Training, network saved')


def task_3_3_3():
    predict_lstmae = LSTMAE_PREDICTOR(input_size_param, hidden_state_size_param, input_size_param, sequence_param - 1)
    train_predict(predict_lstmae)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    dataiter = iter(test_loader)
    stock = dataiter.next()

    fig, ax1 = plt.subplots()
    original = stock.flatten().detach().numpy()
    plt.plot(df.index, original, label='original stock')
    plt.suptitle('Original Vs. Reconstructed Vs. Predicted Stock')
    plt.ylabel('Price')

    predict_lstmae.load_state_dict(torch.load(PATH))

    reconstructed, prediction = predict_lstmae(stock[:, :-1, :])
    reconstructed_stock = reconstructed.flatten().detach().numpy()
    reconstructed_axis = df.index[0:954]
    predicted_stock = prediction.flatten().detach().numpy()
    predicted_axis = df.index[53:1007]
    plt.plot(reconstructed_axis, reconstructed_stock, label='reconstructed stock')
    plt.plot(predicted_axis, predicted_stock, label='predicted stock')
    monthyearFmt = mdates.DateFormatter('%Y %B')
    ax1.xaxis.set_major_formatter(monthyearFmt)
    _ = plt.xticks(rotation=90)
    plt.legend()
    plt.show()


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
    # train(lstmae)

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
# task_3_3_3()
task_3_3_2()
