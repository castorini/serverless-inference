import torch
import torch.nn as nn

import torch.nn.functional as F


class SmPlusPlus(nn.Module):
    def __init__(self, config):
        super(SmPlusPlus, self).__init__()
        output_channel = config.output_channel
        words_dim = config.words_dim
        filter_width = config.filter_width

        n_classes = 2
        ext_feats_size = 4
        input_channel = 1

        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))

        self.dropout = nn.Dropout(config.dropout)
        n_hidden = 2 * output_channel + ext_feats_size

        self.combined_feature_vector = nn.Linear(n_hidden, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_classes)

    def _unsqueeze(self, tensor):
        dim = tensor.size()
        return tensor.view(dim[0], 1, dim[1], dim[2])

    def forward(self, x_question, x_answer, x_ext):
        question = self._unsqueeze(x_question)
        answer = self._unsqueeze(x_answer)  # (batch, 1, sent_len, embed_dim)
        x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling

        # append external features and feed to fc
        x.append(x_ext)
        x = torch.cat(x, 1)

        x = F.tanh(self.combined_feature_vector(x))
        x = self.dropout(x)
        x = self.hidden(x)
        return x
