#Created by Adam Goldbraikh - Scalpel Lab Technion
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.utils.rnn import pack_padded_sequence

class MT_RNN_SlowFast(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_classes_list, bidirectional, dropout, num_layers=2, freq=[1,2]):
        super(MT_RNN_SlowFast, self).__init__()

        assert len(freq) >= 1
        assert num_layers >= 1

        self.rnns = []
        self.frequencies = freq
        self.num_freqs = len(freq)
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        if rnn_type == "GRU":
            for freq in self.frequencies:
                GRU_list = []
                dims = [input_dim // 2 if bidirectional else input_dim] + [hidden_dim for layer_i in range(num_layers)]
                for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                    # print((in_dim * 2 if bidirectional else in_dim, out_dim))
                    GRU_list.append(
                        nn.GRU(in_dim * 2 if bidirectional else in_dim, out_dim, batch_first=True,
                               bidirectional=bidirectional,
                               num_layers=1))
                GRU = nn.ModuleList(GRU_list)
                self.rnns.append(GRU)
            self.rnns = nn.ModuleList(self.rnns)
        else:
            raise NotImplemented
        # The linear layer that maps from hidden state space to tag space
        self.output_heads = nn.ModuleList([copy.deepcopy(
            nn.Linear(hidden_dim * 2*self.num_freqs if bidirectional else hidden_dim * self.num_freqs, num_classes_list[s]) )
                                    for s in range(len(num_classes_list))])

    def forward(self, rnn_inpus, lengths):
        outputs=[]
        freq_outputs = []
        rnn_inpus = rnn_inpus.permute(0, 2, 1)
        rnn_inpus = self.dropout(rnn_inpus)

        #Define different RNNs based on the sampling frequencies
        for i in range(self.num_freqs):
            lengths_after_sampling = torch.ceil(lengths / self.frequencies[i]).to(device=lengths.device)
            sampled_input = rnn_inpus[:, ::self.frequencies[i], :].to(device=rnn_inpus.device)
            sampled_packed_input = pack_padded_sequence(sampled_input, lengths=lengths_after_sampling, batch_first=True,
                                                        enforce_sorted=False)
            temp_output = sampled_packed_input
            for j, layer_j in enumerate(self.rnns[i]):
                temp_output, _ = layer_j(temp_output)
            unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(temp_output, padding_value=-1, batch_first=True)
            upsample = nn.Upsample(scale_factor=self.frequencies[i])
            freq_output = upsample(unpacked_output.permute(0, 2, 1))
            freq_outputs.append(freq_output[:, :, :rnn_inpus.shape[1]])

        unpacked_rnn_out = torch.cat(freq_outputs, dim=1)
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        for output_head in self.output_heads:
            outputs.append(output_head(unpacked_rnn_out.permute(0, 2, 1)).permute(0, 2, 1))
        return outputs


class MT_RNN_SlowFast2(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_classes_list, bidirectional, dropout, num_layers=2, freq=[1,2]):
        super(MT_RNN_SlowFast2, self).__init__()

        assert len(freq)>0

        self.rnns = []
        self.frequencies = freq
        self.num_freqs = len(freq)
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # if rnn_type == "LSTM":
        #     self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
        #                             num_layers=num_layers)
        if rnn_type == "GRU":
            for freq in self.frequencies:
                self.rnns.append(nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, num_layers=num_layers))
            self.rnns=nn.ModuleList(self.rnns)
        else:
            raise NotImplemented
        # The linear layer that maps from hidden state space to tag space
        self.output_heads = nn.ModuleList([copy.deepcopy(
            nn.Linear(hidden_dim * 2*self.num_freqs if bidirectional else hidden_dim * self.num_freqs, num_classes_list[s]) )
                                    for s in range(len(num_classes_list))])

    def forward(self, rnn_inpus, lengths):
        outputs=[]
        freq_outputs = []
        rnn_inpus = rnn_inpus.permute(0, 2, 1)
        rnn_inpus = self.dropout(rnn_inpus)

        #Define different RNNs based on the sampling frequencies
        outputs = []
        for i in range(self.num_freqs):
            lengths_after_sampling = torch.ceil(lengths / self.frequencies[i]).to(device=lengths.device)
            sampled_input = rnn_inpus[:, ::self.frequencies[i], :].to(device=rnn_inpus.device)
            sampled_packed_input = pack_padded_sequence(sampled_input, lengths=lengths_after_sampling, batch_first=True,
                                                        enforce_sorted=False)
            rnn_output, _ = self.rnns[i](sampled_packed_input)
            unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, padding_value=-1, batch_first=True)
            upsample = nn.Upsample(scale_factor=self.frequencies[i])
            freq_output = upsample(unpacked_output.permute(0, 2, 1))
            freq_outputs.append(freq_output[:, :, :rnn_inpus.shape[1]])

        unpacked_rnn_out = torch.cat(freq_outputs, dim=1)
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        for output_head in self.output_heads:
            outputs.append(output_head(unpacked_rnn_out.permute(0, 2, 1)).permute(0, 2, 1))
        return outputs

        #unpacked_rnn_out = torch.cat(unpacked_rnn_out1, unpacked_rnn_out2)

class MT_RNN_dp(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_classes_list, bidirectional, dropout,num_layers=2):
        super(MT_RNN_dp, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                                 num_layers=num_layers)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                                 num_layers=num_layers)
        else:
            raise NotImplemented
        # The linear layer that maps from hidden state space to tag space
        self.output_heads = nn.ModuleList([copy.deepcopy(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes_list[s]) )
                                 for s in range(len(num_classes_list))])


    def forward(self, rnn_inpus, lengths):
        outputs=[]
        rnn_inpus = rnn_inpus.permute(0, 2, 1)
        rnn_inpus=self.dropout(rnn_inpus)

        packed_input = pack_padded_sequence(rnn_inpus, lengths=lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_input)

        unpacked_rnn_out, unpacked_rnn_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, padding_value=-1, batch_first=True)
        # flat_X = torch.cat([unpacked_ltsm_out[i, :lengths[i], :] for i in range(len(lengths))])
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        for output_head in self.output_heads:
            outputs.append(output_head(unpacked_rnn_out).permute(0, 2, 1))
        return outputs


