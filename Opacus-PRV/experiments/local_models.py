import torch
import torch.nn as nn
import torch.nn.functional as F

from opacus.layers import DPLSTM


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x


class CIFAR10_CNN(nn.Module):

    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        print(f"building model!")
        self.build(input_norm, **kwargs)

    def build(self,
              input_norm=None,
              num_groups=None,
              bn_stats=None,
              size=None,
              in_channel=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups,
                                         self.in_channels,
                                         affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                            act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


#
class CNN_Adult(nn.Module):  # THIS IS A FCN

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(123, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        #        x = x.view(-1, 123)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "CNN_Adult"


class LSTMNet(nn.Module):

    def __init__(self, vocab_size=10_000, **_):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = nn.Embedding(vocab_size + 4, 100)
        #self.lstm = nn.LSTM(100, 100)
        self.lstm = DPLSTM(100, 100)

        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        # x: batch_size, seq_len
        x = self.emb(x)  # batch_size, seq_len, embed_dim
        x = x.transpose(0, 1)  # seq_len, batch_size, embed_dim
        x, _ = self.lstm(x)  # seq_len, batch_size, lstm_dim
        x = x.mean(0)  # batch_size, lstm_dim
        x = self.fc1(x)  # batch_size, fc_dim
        return x


class IMDB_RNN(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):

        #text = [sent len, batch size]

        embedded = self.embedding(text)

        #embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


if False:

    class BasicNet(nn.Module):

        def __init__(self, num_features, num_classes):
            super().__init__()
            self.num_features = num_features
            self.num_classes = num_classes
            self.layers = 0

            self.lin1 = torch.nn.Linear(self.num_features, 150)
            self.lin2 = torch.nn.Linear(50, 50)
            self.lin3 = torch.nn.Linear(50, 50)

            self.lin4 = torch.nn.Linear(150, 150)

            self.lin5 = torch.nn.Linear(50, 50)
            self.lin6 = torch.nn.Linear(50, 50)
            self.lin10 = torch.nn.Linear(150, self.num_classes)

            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(0.25)

        def forward(self, xin):
            self.layers = 0

            x = F.relu(self.lin1(xin))
            self.layers += 1

            #x = F.relu(self.lin2(x))
            #self.layers += 1
            for y in range(8):
                x = F.relu(self.lin4(x))
                self.layers += 1

            x = self.dropout(x)

            x = F.relu(self.lin10(x))
            self.layers += 1
            return x
