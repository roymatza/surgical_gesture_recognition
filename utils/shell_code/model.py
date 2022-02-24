#Created by Adam Goldbraikh - Scalpel Lab Technion
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.utils.rnn import pack_padded_sequence

class MT_RNN_SlowFast(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_classes_list, bidirectional, dropout, num_layers=2,
                 freq=[1,2], hidden_dim_ratio=None):
        super(MT_RNN_SlowFast, self).__init__()

        assert len(freq) >= 1
        assert num_layers >= 1

        self.rnns = []
        self.frequencies = freq
        self.num_freqs = len(freq)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fusion = False#True
        self.dropout = torch.nn.Dropout(dropout)

        if hidden_dim_ratio == None:
            self.hidden_dim_ratio = [1] * self.num_freqs
        else:
            assert len(hidden_dim_ratio) == self.num_freqs
            self.hidden_dim_ratio = hidden_dim_ratio
        self.hidden_dims = [int(hidden_dim * freqs_hidden_dim_ratio) for freqs_hidden_dim_ratio in self.hidden_dim_ratio]

        if rnn_type == "GRU":
            for freq, hidden_d in zip(self.frequencies, self.hidden_dims):
                GRU_list = []
                #Seperation of layers
                dims = [input_dim // 2 if bidirectional else input_dim]\
                       + [hidden_d for layer_i in range(num_layers)]
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
        # self.output_heads = nn.ModuleList([copy.deepcopy(
        #     nn.Linear(hidden_dim * 2*self.num_freqs if bidirectional else hidden_dim * self.num_freqs, num_classes_list[s]) )
        #                             for s in range(len(num_classes_list))])
        self.output_heads = nn.ModuleList([copy.deepcopy(
            nn.Linear(sum(self.hidden_dims) * 2 if bidirectional else sum(self.hidden_dims), num_classes_list[s]) )
                                    for s in range(len(num_classes_list))])

    def forward(self, rnn_inpus, lengths):
        rnn_inpus = rnn_inpus.permute(0, 2, 1)
        rnn_inpus = self.dropout(rnn_inpus)
        outputs =[]

        layer_outputs = []
        #forward pass by layers
        for layer in range(self.num_layers):
            #apply lateral connections between paths
            for i, rnn in enumerate(self.rnns):
                curr_layer = rnn[layer]
                if layer==0:
                    #define different RNNs based on the sampling frequencies
                    sampled_packed_input = self.create_packed_sampled_input(rnn_inpus, self.frequencies[i], lengths)
                    layer_output, _ = curr_layer(sampled_packed_input)
                    layer_outputs.append(layer_output)
                    continue               
                
                layer_output, _ = curr_layer(layer_outputs[i])
                layer_outputs[i] = layer_output
                if self.fusion and i>0:
                    fast_output = layer_outputs[i-1] #if layer_outputs[i].data.shape[1] > layer_outputs[i-1].data.shape[1] else layer_outputs[i-1]
                    slow_output = layer_outputs[i] #if layer_outputs[i].data.shape[1] <= layer_outputs[i-1].data.shape[1] else layer_outputs[i-1]
                    lengths_slow_output = torch.ceil(lengths / self.frequencies[i])
                    
                    #update slow path
                    layer_outputs[i] = self.fuse_outputs(slow_output, fast_output, lengths_slow_output)
        
        #upsampling the processed sequences
        for l_i, l_out  in enumerate(layer_outputs):
            layer_outputs[l_i] = self.create_unpacked_upsampled_output(l_out, self.frequencies[l_i], seq_length=rnn_inpus.shape[1])

        #concatenating final result (channel dimension)
        unpacked_rnn_out = torch.cat(layer_outputs, dim=-1)
        
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        for output_head in self.output_heads:
            outputs.append(output_head(unpacked_rnn_out).permute(0,2,1))
        return outputs
        
    def create_packed_sampled_input(self, input, sample_ratio, seq_lengths):
        '''creating downsampled input packed for RNN'''
        lengths_after_sampling = torch.ceil(seq_lengths / sample_ratio).to(device=seq_lengths.device)
        sampled_input = input[:, ::sample_ratio, :].to(device=input.device)
        
        return pack_padded_sequence(sampled_input, lengths=lengths_after_sampling, batch_first=True,
                                                    enforce_sorted=False)
    
    def create_unpacked_upsampled_output(self, output, upsample_ratio, seq_length):
        '''creating the output in an unpacked manner, upsampled to the original frequency'''
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=-1, batch_first=True)
        upsample = nn.Upsample(scale_factor=upsample_ratio)
        output_upsampled = upsample(unpacked_output.permute(0, 2, 1)).permute(0,2,1)
        return output_upsampled[:, :seq_length, :]
         
        
    def fuse_outputs(self, out_small, out_large, seq_lengths_small, method='sample', op='cat'):
        '''Applying fusion of two different output tensors, by whether summation or concatenation'''
        '''Assuming input of shape NxSxD, where N is batchsize, S is sequence length and D is the hidden dimension'''
        
        #unpack and pad tensors
        out_small, _ = torch.nn.utils.rnn.pad_packed_sequence(out_small, padding_value=-1, batch_first=True)
        out_large, _ = torch.nn.utils.rnn.pad_packed_sequence(out_large, padding_value=-1, batch_first=True)
        
        #modify tensors for fusion
        if method=='time2channel':
            #WARNING: changes number of output channels
            out_modified = out_large.view(out_large.shape[0],out_small.shape[1],-1)
        if method=='sample':
            sample_ratio = torch.ceil(torch.Tensor([out_large.shape[1]/out_small.shape[1]])).to(dtype=torch.long)
            out_modified = out_large[:,::sample_ratio,:] #now small and large tensors has the same sequence length

        #apply fusion operation
        if op=='cat':
            result= torch.cat((out_modified,out_small),dim=1)
        if op=='sum':
            result= out_modified+out_small
        
        return torch.nn.utils.rnn.pack_padded_sequence(result, seq_lengths_small, batch_first=True, enforce_sorted=False)



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


