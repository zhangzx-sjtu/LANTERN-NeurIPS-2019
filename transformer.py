import torch
import torch.nn as nn
import numpy as np


class FeedForwardLayer(nn.Module):

    def __init__(self, model_dim, inner_dim, dropout):

        super(FeedForwardLayer, self).__init__()
        self.linear_in_layer = nn.Linear(model_dim, inner_dim)
        self.activation = nn.ReLU()
        self.linear_out_layer = nn.Linear(inner_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, single_data):
        """ single_data: (batch, 1, model_dim), hidden state in a single step """
        inner_state = self.dropout(self.linear_in_layer(single_data))
        middle_state = self.activation(inner_state)
        output_state = self.dropout(self.linear_out_layer(middle_state))
        output = self.layer_norm(output_state)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, model_dim, d_q, d_k, d_v, cuda_id, dropout):

        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.model_dim = model_dim
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.cuda_id = cuda_id
        self.discount = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.weight_q = nn.Parameter(torch.randn(model_dim, heads * d_q)).cuda(self.cuda_id)
        self.weight_k = nn.Parameter(torch.randn(model_dim, heads * d_k)).cuda(self.cuda_id)
        self.weight_v = nn.Parameter(torch.randn(model_dim, heads * d_v)).cuda(self.cuda_id)
        self.fc_layer = nn.Linear(heads * d_v, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, data_input, index):
        """ data_input: size: (batch, max_seq_len, model_dim) """
        size = data_input.size()

        involved_data = data_input[:, 0:index+1, :]

        involved_q = torch.matmul(involved_data, self.weight_q)
        involved_k = torch.matmul(involved_data, self.weight_k)
        involved_v = torch.matmul(involved_data, self.weight_v)

        target_q = involved_q[:, index:index+1, :]

        product_weight = target_q * involved_k
        concate_value = torch.zeros(size[0], 1, self.heads * self.d_v).cuda(self.cuda_id)

        for i in range(self.heads):
            weight_matrix = product_weight[:, :, (i * self.d_q):((i + 1) * self.d_q)]
            value_matrix = involved_v[:, :, (i * self.d_v):((i + 1) * self.d_v)]
            sum_weight = torch.sum(weight_matrix, 2)
            softmax_weight = torch.softmax(sum_weight/self.discount, 1).unsqueeze(2)
            weighted_value = value_matrix * softmax_weight
            final_head_value = torch.sum(weighted_value, 1).unsqueeze(1)
            concate_value[:, 0:1, (i * self.d_v):((i + 1) * self.d_v)] = final_head_value

        integrate_value = self.fc_layer(concate_value)
        final_value = self.layer_norm(integrate_value)

        return final_value


class Transformer(nn.Module):

    def __init__(self, heads, model_dim, inner_dim, d_q, d_k, d_v, cuda_id, dropout):
        super(Transformer, self).__init__()
        self.feed_forward = FeedForwardLayer(model_dim, inner_dim, dropout)
        self.multi_head_attn = MultiHeadAttention(heads, model_dim, d_q, d_k, d_v, cuda_id, dropout)

    def forward(self, data_input, index):
        multi_attn_output = self.multi_head_attn(data_input, index)
        layer_output = self.feed_forward(multi_attn_output)
        return layer_output


if __name__ == "__main__":
    """ test for FeedForwardLayer """
    # test = torch.rand(3, 1, 4)
    # model = FeedForwardLayer(4, 5)
    # res = model.forward(test)
    # print(res)
    """ test for MultiHeadAttention """
    # test = torch.ones(3, 4, 5).cuda()
    # model = MultiHeadAttention(4, 5, 6, 6, 8, 0).cuda()
    # res = model.forward(test, 2)
    # print(res)
    """ test for transformer layer"""
    test = torch.ones(3, 4, 5).cuda()
    model = Transformer(4, 5, 10, 6, 6, 8, 0, 0).cuda()
    res = model.forward(test, 2)
    print(res)
    torch.mean(res).backward()