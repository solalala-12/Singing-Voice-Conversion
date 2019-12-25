import torch.nn as nn
import torch
import numpy as np


class Wavenet(nn.Module):
    def __init__(self, n_class, hidden_channels, cond_channels, n_repeat, n_layer, device):
        super(Wavenet, self).__init__()
        self.n_class = n_class
        self.hidden_channels = hidden_channels

        self.input_conv = nn.Conv1d(in_channels=n_class,
                                    out_channels=hidden_channels,
                                    kernel_size=1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.cond_filter_convs = nn.ModuleList()
        self.cond_gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.dilations = []
        self.pad = []

        for _ in range(n_repeat):
            for i in range(n_layer):
                dilation = 2**i
                self.dilations.append(dilation)
                self.pad.append(nn.ConstantPad1d((dilation, 0), 0))
                self.filter_convs.append(nn.Conv1d(in_channels=hidden_channels,
                                                   out_channels=hidden_channels,
                                                   kernel_size=2,
                                                   dilation=dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=hidden_channels,
                                                 out_channels=hidden_channels,
                                                 kernel_size=2,
                                                 dilation=dilation))
                self.cond_filter_convs.append(nn.Conv1d(in_channels=cond_channels,
                                                        out_channels=hidden_channels,
                                                        kernel_size=2,
                                                        dilation=dilation))
                self.cond_gate_convs.append(nn.Conv1d(in_channels=cond_channels,
                                                      out_channels=hidden_channels,
                                                      kernel_size=2,
                                                      dilation=dilation))
                self.residual_convs.append(nn.Conv1d(in_channels=hidden_channels,
                                                     out_channels=hidden_channels,
                                                     kernel_size=1))
                self.skip_convs.append(nn.Conv1d(in_channels=hidden_channels,
                                                 out_channels=hidden_channels,
                                                 kernel_size=1))


        self.output_conv1 = nn.Conv1d(in_channels=hidden_channels,
                                      out_channels=hidden_channels,
                                      kernel_size=1)
        self.output_conv2 = nn.Conv1d(in_channels=hidden_channels,
                                      out_channels=n_class,
                                      kernel_size=1)
        self.relu = nn.ReLU()


        self.max_dilation = max(self.dilations)
        self.device = device
        self.to(device)

    def forward(self, x, cond):
        x = self.input_conv(x)
        skip = torch.zeros((x.shape[0], self.hidden_channels, x.shape[2]),
                           dtype=torch.float,
                           device=self.device)

        for i, dilation in enumerate(self.dilations):
            padded_x = self.pad[i](x)
            padded_cond = self.pad[i](cond)
            fx = self.filter_convs[i](padded_x)
            gx = self.gate_convs[i](padded_x)
            fc = self.cond_filter_convs[i](padded_cond)
            gc = self.cond_gate_convs[i](padded_cond)
            z = torch.tanh(fx+fc)*torch.sigmoid(gx+gc)

            skip += self.skip_convs[i](z)
            x += self.residual_convs[i](z)

        y = self.relu(skip)
        y = self.output_conv1(y)
        y = self.relu(y)
        y = self.output_conv2(y)
        return y

    def generate(self, cond, max_length):
        x = torch.zeros((cond.shape[0], self.n_class, cond.shape[2]),
                            dtype=torch.float,
                            device=self.device)
        x[:, np.random.randint(self.n_class, size=cond.shape[0]), 0] = 1
        res = []
        for i in range(min(cond.shape[2], max_length)):
            y = self.forward(x[:, :, :i+1], cond[:, :, :i+1])
            if i+1 < cond.shape[1]:
                out = np.argmax(y[:, :, i].cpu().detach().numpy(), axis=1)
                x[:, out, i+1] = 1
                res.append(out)
        return np.column_stack(res)