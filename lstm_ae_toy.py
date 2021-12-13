import torch
import torch.nn as nn
import torch.optim as optim

from synthetic_data_generator import synthetic_data_gen


class LSTMAE(nn.Module):
  def __init__(self, m, k, l):
        super().__init__()
        self.lstm_enc = nn.LSTM(m, k, batch_first=True)
        self.lstm_dec = nn.LSTM(k, l, batch_first=True)

  def forward(self, x):
    # print("x", x.shape)
    output, (_, _) = self.lstm_enc(x)
    # print("output_enc", output.shape)
    _, _, k = output.size()
    # print("k", k)
    to_repeat = output[:, -1, :]
    # print("to_repeat", to_repeat.shape)
    to_repeat = to_repeat[:, None, :]
    # print("to_repeat reshape", to_repeat.shape)
    z = to_repeat.repeat(1, t, 1)
    # print("z", z.shape)
    output_dec , (_, _) = self.lstm_dec(z)
    # print("output_dec", output_dec.shape)
    return output_dec

lstmae = LSTMAE(1, 40, 1)

sequences = 10000
t = 50

criterion = nn.MSELoss()
optimizer = optim.SGD(lstmae.parameters(), lr=0.0001, momentum=0.9)


data = synthetic_data_gen()
loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, num_workers=0)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
    # inputs, labels = data
        # print("data shape", data.shape)
        # print("data", data)
        # first_3 = data[:3, :]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # tensor_first_3 = torch.tensor(first_3)
        outputs = lstmae(data)
        # print("output", outputs.shape)
        # print("first_3", tensor_first_3.shape)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
          print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 200))
          running_loss = 0.0


print('Finished Training')