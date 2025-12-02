import torch
import torch.nn as nn
class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        input_dim = args.enc_in
        hidden_dim = args.d_model
        output_dim = args.c_out
        num_layers = args.e_layers
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                        num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
    
