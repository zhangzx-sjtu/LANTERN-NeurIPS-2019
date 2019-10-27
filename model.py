import torch
import torch.nn as nn

from generator import Generator, RNN_Generator
from discriminator import Discriminator, RNN_Discriminator, Reward
from optimization import D_Loss, PolicyGradient


class MainModel(nn.Module):

    def __init__(self, marker_num, neighbor_num, embed_dim, d_model, d_inner, d_q, d_k, d_v,
                 n_head, candi_size, max_time, beta, cuda_id, K, discount, regular, dropout=0.1):

        super(MainModel, self).__init__()
        self.generator = Generator(marker_num, neighbor_num, embed_dim, d_model, d_inner, d_q, d_k,
                                   d_v, n_head, candi_size, max_time, beta, cuda_id, dropout=dropout)
        self.discriminator = Discriminator(marker_num, embed_dim, d_model, d_inner, d_q, d_k,
                                           d_v, n_head, beta, cuda_id, K, dropout=dropout)

        self.marker_embeddings = nn.Parameter(torch.ones(marker_num, d_model))
        self.d_loss_func = D_Loss(K)
        self.g_loss_func = PolicyGradient(discount, regular, K, cuda_id)
        self.discount = discount
        self.regular = regular
        self.marker_num = marker_num
        self.K = K

    def forward(self, marker_data, time_data, mask_data):
        gen_markers, gen_times, gen_masks, gen_p_neighbor, gen_p_sample = [], [], [], [], []

        for i in range(self.K):
            new_markers, new_times, new_masks, new_p_neighbor, new_p_sample = \
                self.generator.forward(marker_data, time_data, mask_data, self.marker_embeddings)
            gen_markers.append(new_markers)
            gen_times.append(new_times.detach())
            gen_masks.append(new_masks)
            gen_p_neighbor.append(new_p_neighbor)
            gen_p_sample.append(new_p_sample)

        true_reward, true_masks, bogus_reward, bogus_masks = \
            self.discriminator.forward(marker_data, time_data, mask_data,
                                       gen_markers, gen_times, gen_masks, self.marker_embeddings)

        d_loss = self.d_loss_func.forward(true_reward, true_masks, bogus_reward, bogus_masks)
        g_loss = self.g_loss_func.forward(gen_p_neighbor, gen_p_sample, bogus_reward, bogus_masks)

        return d_loss, g_loss


class RNN_Model(nn.Module):

    def __init__(self, marker_num, neighbor_num, embed_dim, d_model, candi_size,
                 max_time, beta, cuda_id, K, discount, regular, dropout=0.1):

        super(RNN_Model, self).__init__()

        self.generator = RNN_Generator(marker_num, neighbor_num, embed_dim, d_model,
                                       candi_size, max_time, beta, cuda_id, dropout=dropout)
        self.discriminator = RNN_Discriminator(marker_num, embed_dim, d_model, beta, cuda_id, K, dropout=dropout)
        self.marker_embeddings = nn.Parameter(torch.ones(marker_num, d_model).cuda(cuda_id))

        self.d_loss_func = D_Loss(K)
        self.g_loss_func = PolicyGradient(discount, regular, K, cuda_id)

        self.discount = discount
        self.regular = regular
        self.marker_num = marker_num
        self.K = K

    def forward(self, marker_data, time_data, mask_data):
        gen_markers, gen_times, gen_masks, gen_p_neighbor, gen_p_sample = [], [], [], [], []
        for i in range(self.K):
            new_markers, new_times, new_masks, new_p_neighbor, new_p_sample = \
                self.generator.forward(marker_data, time_data, mask_data, self.marker_embeddings)
            gen_markers.append(new_markers)
            gen_times.append(new_times.detach())
            gen_masks.append(new_masks)
            gen_p_neighbor.append(new_p_neighbor)
            gen_p_sample.append(new_p_sample)

        true_reward, true_masks, bogus_reward, bogus_masks = \
            self.discriminator.forward(marker_data, time_data, mask_data, gen_markers, gen_times, gen_masks, self.marker_embeddings)

        d_loss = self.d_loss_func.forward(true_reward, true_masks, bogus_reward, bogus_masks)
        g_loss = self.g_loss_func.forward(gen_p_neighbor, gen_p_sample, bogus_reward, bogus_masks)

        return d_loss, g_loss


class PR_Model(nn.Module):

    def __init__(self, marker_num, neighbor_num, embed_dim, d_model, d_inner, d_q, d_k, d_v,
                 n_head, candi_size, max_time, beta, cuda_id, K, discount, regular, dropout=0.1):

        super(PR_Model, self).__init__()
        self.generator = Generator(marker_num, neighbor_num, embed_dim, d_model, d_inner, d_q, d_k,
                                   d_v, n_head, candi_size, max_time, beta, cuda_id, dropout=dropout)
        self.reward_func = Reward()
        self.marker_embeddings = nn.Parameter(torch.ones(marker_num, d_model).cuda(cuda_id))
        self.g_loss_func = PolicyGradient(discount, regular, K, cuda_id)
        self.discount = discount
        self.regular = regular
        self.marker_num = marker_num
        self.K = K

    def forward(self, marker_data, time_data, mask_data):
        gen_markers, gen_times, gen_masks, gen_p_neighbor, gen_p_sample = [], [], [], [], []
        for i in range(self.K):
            new_markers, new_times, new_masks, new_p_neighbor, new_p_sample = \
                self.generator.forward(marker_data, time_data, mask_data, self.marker_embeddings)
            gen_markers.append(new_markers)
            gen_times.append(new_times.detach())
            gen_masks.append(new_masks)
            gen_p_neighbor.append(new_p_neighbor)
            gen_p_sample.append(new_p_sample)

        bogus_rewards, bogus_masks = self.reward_func.forward(marker_data, time_data, mask_data,
                                                              gen_markers, gen_times, gen_masks, self.marker_embeddings)
        g_loss = self.g_loss_func.forward(gen_p_neighbor, gen_p_sample, bogus_rewards, bogus_masks)

        return g_loss


if __name__ == "__main__":
    marker = torch.LongTensor([[1, 2, 3, 5], [2, 4, 3, 1], [0, 0, 3, 3]]).cuda(0)
    time = torch.FloatTensor([[0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7]]).cuda(0)
    mask = torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]]).cuda(0)

    model = MainModel(6, 5, 3, 10, 20, 10, 10, 10, 4, 1.5, 0.3, 0, 3, 0.99, 0.001, 0).cuda(0)
    # model = RNN_Model(6, 5, 3, 10, 1, 0.3, 0, 3, 0.99, 0.001, 0).cuda(0)
    # model = PR_Model(6, 5, 3, 10, 20, 10, 10, 10, 4, 1.5, 0.3, 0, 3, 0.99, 0.001, 0.1).cuda(0)









