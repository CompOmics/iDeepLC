from torch import nn, cat
from torch.nn import init


# Create Network architecture
class MyNet(nn.Module):
    def __init__(self, x_shape, config):
        self.x_shape = x_shape
        super(MyNet, self).__init__()

        #  first cnn layers for amino

        self.l1 = nn.ModuleList()

        self.l1.append(
            nn.Conv1d(in_channels=1, out_channels=config['cnn_channels'], kernel_size=(config['kernel_size'],),
                      stride=(1,), padding=(int((config['kernel_size'] - 1) / 2),)))
        self.l1.append(nn.ELU())

        for layer in range(config['cnn_layers']):
            self.l1.append(nn.Conv1d(in_channels=config['cnn_channels'], out_channels=config['cnn_channels'],
                                     kernel_size=(config['kernel_size'],), stride=(1,),
                                     padding=(int((config['kernel_size'] - 1) / 2),)))
            self.l1.append(nn.ELU())
            self.l1.append(nn.Dropout(p=config['dropout'], inplace=False))

        self.l1.append(nn.Flatten())

        #  second cnn layers for diamino

        self.l2 = nn.ModuleList()

        self.l2.append(
            nn.Conv1d(in_channels=7, out_channels=config['cnn2_channels'], kernel_size=(config['kernel2_size'],),
                      stride=(1,), padding=(int((config['kernel2_size'] - 1) / 2),)))
        self.l2.append(nn.ELU())

        for layer in range(config['cnn2_layers']):
            self.l2.append(nn.Conv1d(in_channels=config['cnn2_channels'], out_channels=config['cnn2_channels'],
                                     kernel_size=(config['kernel2_size'],), stride=(1,),
                                     padding=(int((config['kernel2_size'] - 1) / 2),)))
            self.l2.append(nn.ELU())
            self.l2.append(nn.Dropout(p=config['dropout'], inplace=False))

        self.l2.append(nn.Flatten())

        #  third cnn layers for atoms

        self.l3 = nn.ModuleList()

        self.l3.append(
            nn.Conv1d(in_channels=6, out_channels=config['cnn3_channels'], kernel_size=(config['kernel3_size'],),
                      stride=(1,), padding=(int((config['kernel3_size'] - 1) / 2),)))
        self.l3.append(nn.ELU())

        for layer in range(config['cnn3_layers']):
            self.l3.append(nn.Conv1d(in_channels=config['cnn3_channels'], out_channels=config['cnn3_channels'],
                                     kernel_size=(config['kernel3_size'],), stride=(1,),
                                     padding=(int((config['kernel3_size'] - 1) / 2),)))
            self.l3.append(nn.ELU())
            self.l3.append(nn.Dropout(p=config['dropout'], inplace=False))

        self.l3.append(nn.Flatten())

        #  first fc layer for generale features

        self.l4 = nn.ModuleList()

        self.l4.append(nn.Linear(in_features=self.x_shape[2], out_features=config['fc_out']))
        self.l4.append(nn.ReLU())

        for layer in range(config['fc_layers']):
            self.l4.append(nn.Linear(in_features=config['fc_out'], out_features=config['fc_out']))
            self.l4.append(nn.ReLU())
        self.l4.append(nn.Flatten())

        #  fourth cnn layer for one hot encoder

        self.l5 = nn.ModuleList()

        self.l5.append(
            nn.Conv1d(in_channels=20, out_channels=config['cnn4_channels'], kernel_size=(config['kernel4_size'],),
                      stride=(1,), padding=(int((config['kernel4_size'] - 1) / 2),)))
        self.l5.append(nn.ELU())
        self.l5.append(nn.Dropout(p=config['dropout'], inplace=False))

        for layer in range(config['cnn4_layers']):
            self.l5.append(nn.Conv1d(in_channels=config['cnn4_channels'], out_channels=config['cnn4_channels'],
                                     kernel_size=(config['kernel4_size'],), stride=(1,),
                                     padding=(int((config['kernel4_size'] - 1) / 2),)))
            self.l5.append(nn.ELU())
            self.l5.append(nn.Dropout(p=config['dropout'], inplace=False))

        self.l5.append(nn.Flatten())

        #  final FC layer for concatenating all layers

        self.l6 = nn.ModuleList()

        self.l6.append(nn.Linear(in_features=(
                ((config['cnn_channels']) + config['cnn3_channels'] + config['cnn4_channels']) * self.x_shape[2] +
                config['fc_out'] * 7 + config['cnn2_channels'] * int(self.x_shape[2] / 2)),
            out_features=int(config['fc2_out'])))
        self.l6.append(nn.ELU())

        for layer in range(config['fc2_layers']):
            self.l6.append(nn.Linear(in_features=config['fc2_out'], out_features=config['fc2_out']))
            self.l6.append(nn.ELU())

        self.l6.append(nn.Linear(in_features=config['fc2_out'], out_features=int(config['fc2_out'])))
        self.l6.append(nn.ReLU())
        self.l6.append(nn.Linear(in_features=int(config['fc2_out']), out_features=int(config['fc2_out'])))
        self.l6.append(nn.ReLU())
        self.l6.append(nn.Linear(in_features=int(config['fc2_out']), out_features=1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:, :1, :]
        x2 = x[:, 1:8, :int(self.x_shape[2] / 2)]
        x3 = x[:, 8:14, :]
        x4 = x[:, 14:21, :]
        x5 = x[:, 21:, :]

        for layer in self.l1:
            x1 = layer(x1)

        for layer in self.l2:
            x2 = layer(x2)

        for layer in self.l3:
            x3 = layer(x3)

        for layer in self.l4:
            x4 = layer(x4)

        for layer in self.l5:
            x5 = layer(x5)

        x6 = cat((x1, x2, x3, x4, x5), 1)
        for layer in self.l6:
            x6 = layer(x6)

        return x6
