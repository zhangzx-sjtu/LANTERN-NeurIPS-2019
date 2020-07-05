import torch
import torch.nn as nn


class D_Loss(nn.Module):

    def __init__(self, K):
        super(D_Loss, self).__init__()
        self.K = K

    def forward(self, real_rewards, real_masks, fake_rewards, fake_masks):
        # Calculate real sequences loss
        real_lengths = torch.sum(real_masks, 1)
        real_loss = torch.mean(torch.sum(torch.log(real_rewards + 1 - real_masks) * real_masks, 1) / real_lengths)
        fake_loss = []

        # Calculate fake sequences loss
        list_size = len(fake_masks)
        for i in range(list_size):
            fake_length = torch.sum(fake_masks[i], 1)
            fake_loss.append(torch.sum(torch.log(1 - fake_rewards[i]) * fake_masks[i], 1) / fake_length)

        final_fake_loss = torch.mean(sum(fake_loss)/self.K)
        loss = - (final_fake_loss + real_loss)
        return loss


class PolicyGradient(nn.Module):

    def __init__(self, discount, regular, K, cuda_id):
        super(PolicyGradient, self).__init__()
        self.discount = discount
        self.regular = regular
        self.K = K
        self.cuda_id = cuda_id

    def get_discount_matrix(self, size_x, size_y):
        matrix = self.discount * torch.ones(size_x, 1).cuda(self.cuda_id)
        new_column = self.discount * torch.ones(size_x, 1).cuda(self.cuda_id)
        for i in range(size_y - 1):
            new_column = self.discount * new_column
            matrix = torch.cat((matrix, new_column), 1)
        return matrix

    def forward(self, prob_neighbor, prob_sample, rewards, masks):
        # all input arguments are lists of size K, within each (batch, max_seq_len)
        list_size = len(prob_sample)
        size = prob_sample[0].size()
        dis_matrix = self.get_discount_matrix(size[0], size[1])
        # print("dis_matrix", dis_matrix)
        for i in range(list_size):
            lengths = torch.sum(masks[i], 1).unsqueeze(-1)
            # print(lengths)
            rewards_detach = (rewards[i] - torch.sum(rewards[i], 1).unsqueeze(-1) / lengths).detach() * masks[i]
            prob_output = torch.log(prob_neighbor[i] * prob_sample[i] + 1 - masks[i] + 1e-8)
            prob_output_detach = prob_output.detach()
            if i == 0:
                target = torch.sum(dis_matrix * prob_output * rewards_detach * masks[i], 1)
                regularization = - self.regular * torch.sum(prob_output_detach * prob_output * masks[i], 1)
            else:
                target = target + torch.sum(dis_matrix * prob_output * rewards_detach * masks[i], 1)
                regularization = regularization - self.regular * torch.sum(prob_output_detach * prob_output * masks[i], 1)
        
        target = target + regularization
        loss = - torch.mean(target)
        return loss


if __name__ == "__main__":
    real_rewards = torch.FloatTensor([[0.1, 0.3, 0.5, 0], [0.3, 0.4, 0.7, 0.9], [0.3, 0.4, 0.5, 0]]).cuda(0)
    real_masks = torch.FloatTensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]]).cuda(0)

    fake_rewards_1 = torch.FloatTensor([[0.1, 0.3, 0.5, 0], [0.3, 0.4, 0.7, 0.9], [0.3, 0.4, 0.5, 0]]).cuda(0)
    fake_rewards_2 = torch.FloatTensor([[0.1, 0.3, 0.5, 0], [0.3, 0.4, 0.7, 0.9], [0.3, 0.4, 0.5, 0]]).cuda(0)

    fake_masks_1 = torch.FloatTensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]]).cuda(0)
    fake_masks_2 = torch.FloatTensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]]).cuda(0)

    fake_rewards = [fake_rewards_1, fake_rewards_2]
    fake_masks = [fake_masks_1, fake_masks_2]

    d_loss = D_Loss(2)
    res = d_loss.forward(real_rewards, real_masks, fake_rewards, fake_masks).cuda(0)
    print(res)
    torch.mean(res).backward()



