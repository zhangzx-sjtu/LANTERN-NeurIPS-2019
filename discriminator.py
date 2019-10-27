import torch
import torch.nn as nn

from transformer import Transformer


class Discriminator(nn.Module):

    def __init__(self, marker_num, embed_dim, d_model, d_inner, d_q,
                 d_k, d_v, n_head, beta, cuda_id, K, dropout=0.1):

        super(Discriminator, self).__init__()
        # Modules
        self.encoder = Transformer(n_head, d_model, d_inner, d_q, d_k, d_v, cuda_id, dropout)
        self.output_linear = nn.Linear(d_model, 1)
        self.time_embed_linear = nn.Linear(1, d_model)
        self.embed_linear = nn.Linear(embed_dim, d_model)
        self.embed_ac = nn.LeakyReLU()
        # Constants
        self.embed_dim = embed_dim
        self.marker_num = marker_num
        self.beta = beta
        self.K = K
        self.d_model = d_model
        self.cuda_id = cuda_id
        # Indices
        self.real_index = 0
        self.fake_index = 0

    def get_time_embedding(self, time):
        time_embed = self.time_embed_linear(time)
        time_embed_res = time_embed.unsqueeze(1)
        return time_embed_res

    def get_marker_embedding(self, marker, embedding):
        marker_embed_res = embedding[marker]
        return marker_embed_res

    def get_embedding(self, marker, time, embedding):
        time_vector = self.get_time_embedding(time)
        marker_vector = self.get_marker_embedding(marker, embedding)
        vector = marker_vector + self.beta * time_vector
        return vector

    def real_forward(self, real_marker, real_time, real_mask, embedding):
        self.real_index = 0
        real_size = real_time.size()
        real_data_input = torch.zeros(real_size[0], 0, self.d_model).cuda(self.cuda_id)
        real_reward = torch.zeros(real_size[0], 0).cuda(self.cuda_id)

        while self.real_index <= real_size[1] - 1:
            real_new_vector = self.get_embedding(real_marker[:, self.real_index: self.real_index + 1],
                                                 real_time[:, self.real_index: self.real_index + 1],
                                                 embedding)
            model_dim_vector = self.embed_ac(self.embed_linear(real_new_vector))
            real_data_input = torch.cat((real_data_input, model_dim_vector), 1)
            real_intensity = self.encoder.forward(real_data_input, self.real_index)
            step_real_reward = torch.sigmoid(self.output_linear(real_intensity.squeeze(1)))
            real_reward = torch.cat((real_reward, step_real_reward), 1)
            self.real_index += 1

        real_reward = real_reward * real_mask
        return real_reward

    def fake_forward(self, fake_marker, fake_time, fake_mask, embedding):
        self.fake_index = 0
        fake_size = fake_time.size()
        fake_data_input = torch.zeros(fake_size[0], 0, self.d_model).cuda(self.cuda_id)
        fake_reward = torch.zeros(fake_size[0], 0).cuda(self.cuda_id)

        while self.fake_index <= fake_size[1] - 1:
            fake_new_vector = self.get_embedding(fake_marker[:, self.fake_index: self.fake_index + 1],
                                                 fake_time[:, self.fake_index: self.fake_index + 1],
                                                 embedding)
            model_dim_vector = self.embed_ac(self.embed_linear(fake_new_vector))
            fake_data_input = torch.cat((fake_data_input, model_dim_vector), 1)
            fake_intensity = self.encoder.forward(fake_data_input, self.fake_index)
            step_fake_reward = torch.sigmoid(self.output_linear(fake_intensity.squeeze(1)))
            fake_reward = torch.cat((fake_reward, step_fake_reward), 1)
            self.fake_index += 1

        fake_reward = fake_reward * fake_mask
        return fake_reward

    def forward(self, real_marker, real_time, real_mask, fake_marker, fake_time, fake_mask,
                embedding_matrix):
        embedding = embedding_matrix.detach()
        real_rewards = self.real_forward(real_marker, real_time, real_mask, embedding)
        fake_rewards = []

        for i in range(self.K):
            single_fake_r = self.fake_forward(fake_marker[i], fake_time[i], fake_mask[i], embedding)
            fake_rewards.append(single_fake_r)

        return real_rewards, real_mask, fake_rewards, fake_mask


class RNN_Discriminator(nn.Module):

    def __init__(self, marker_num, embed_dim, d_model, beta, cuda_id, K, dropout=0.1):

        super(RNN_Discriminator, self).__init__()
        # Modules
        self.embed_linear = nn.Linear(embed_dim, d_model)
        self.embed_ac = nn.LeakyReLU()
        self.encoder = nn.RNN(d_model, d_model, 1, batch_first=True, dropout=dropout)
        self.time_embed_linear = nn.Linear(1, d_model)
        self.output_linear = nn.Linear(d_model, 1)
        # Constants
        self.marker_num = marker_num
        self.beta = beta
        self.K = K
        self.d_model = d_model
        self.cuda_id = cuda_id
        # Indices
        self.real_index = 0
        self.fake_index = 0

    def get_time_embedding(self, time):
        time_embed = self.time_embed_linear(time)
        time_embed_res = time_embed.unsqueeze(1)
        return time_embed_res

    def get_marker_embedding(self, marker, embedding):
        marker_embed_res = embedding[marker]
        return marker_embed_res

    def get_embedding(self, marker, time, embedding):
        time_vector = self.get_time_embedding(time)
        marker_vector = self.get_marker_embedding(marker, embedding)
        vector = marker_vector + self.beta * time_vector
        return vector

    def real_forward(self, real_marker, real_time, real_mask, embedding):
        self.real_index = 0
        real_size = real_time.size()
        real_data_input = torch.zeros(real_size[0], 0, self.d_model).cuda(self.cuda_id)
        real_reward = torch.zeros(real_size[0], 0).cuda(self.cuda_id)

        real_hidden_state = torch.zeros(1, real_size[0], self.d_model).cuda(self.cuda_id)

        while self.real_index <= real_size[1] - 1:
            real_new_vector = self.get_embedding(real_marker[:, self.real_index: self.real_index + 1],
                                                 real_time[:, self.real_index: self.real_index + 1], embedding)
            model_dim_vector = self.embed_ac(self.embed_linear(real_new_vector))
            real_data_input = torch.cat((real_data_input, model_dim_vector), 1)
            rnn_output = self.encoder(real_data_input[:, self.real_index: self.real_index + 1, :], real_hidden_state)
            real_intensity = rnn_output[0]
            real_hidden_state = rnn_output[1]
            step_real_reward = torch.sigmoid(self.output_linear(real_intensity.squeeze(1)))
            real_reward = torch.cat((real_reward, step_real_reward), 1)
            self.real_index += 1

        real_reward = real_reward * real_mask
        return real_reward

    def fake_forward(self, fake_marker, fake_time, fake_mask, embedding):
        self.fake_index = 0
        fake_size = fake_time.size()
        fake_data_input = torch.zeros(fake_size[0], 0, self.d_model).cuda(self.cuda_id)
        fake_reward = torch.zeros(fake_size[0], 0).cuda(self.cuda_id)

        fake_hidden_state = torch.zeros(1, fake_size[0], self.d_model).cuda(self.cuda_id)

        while self.fake_index <= fake_size[1] - 1:
            fake_new_vector = self.get_embedding(fake_marker[:, self.fake_index: self.fake_index + 1],
                                                 fake_time[:, self.fake_index: self.fake_index + 1], embedding)
            model_dim_vector = self.embed_ac(self.embed_linear(fake_new_vector))
            fake_data_input = torch.cat((fake_data_input, model_dim_vector), 1)
            rnn_output = self.encoder(fake_data_input[:, self.fake_index: self.fake_index + 1, :], fake_hidden_state)
            fake_intensity = rnn_output[0]
            fake_hidden_state = rnn_output[1]
            step_fake_reward = torch.sigmoid(self.output_linear(fake_intensity.squeeze(1)))
            fake_reward = torch.cat((fake_reward, step_fake_reward), 1)
            self.fake_index += 1

        fake_reward = fake_reward * fake_mask
        return fake_reward

    def forward(self, real_marker, real_time, real_mask, fake_marker, fake_time, fake_mask, embedding_matrix):
        embedding = embedding_matrix.detach()
        real_rewards = self.real_forward(real_marker, real_time, real_mask, embedding)
        fake_rewards = []

        for i in range(self.K):
            single_fake_r = self.fake_forward(fake_marker[i], fake_time[i], fake_mask[i], embedding)
            fake_rewards.append(single_fake_r)

        return real_rewards, real_mask, fake_rewards, fake_mask


class Reward(nn.Module):

    def __init__(self):
        super(Reward, self).__init__()

    def forward(self, real_marker, real_time, real_mask, fake_marker, fake_time, fake_mask, embedding):
        rewards = []

        fake_size = len(fake_time)
        for i in range(fake_size):
            marker_reward = (fake_marker[i] == real_marker).float() * real_mask
            time_reward = (fake_time[i] - real_time) * (fake_time[i] - real_time) * real_mask
            reward = torch.sigmoid(marker_reward + time_reward)
            rewards.append(reward)

        return rewards, fake_mask


if __name__ == "__main__":
    d = RNN_Discriminator(6, 5, 8, 0.3, 0, 2, 0.1).cuda(0)


