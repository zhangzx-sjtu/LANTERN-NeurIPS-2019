import torch


class Small_Network_Evaluator(object):

    def __init__(self, marker_size, neg_size, test_size, adjacent):

        self.marker_size = marker_size
        self.neg_size = neg_size
        self.test_size = test_size

        self.adjacent = adjacent
        self.neg_adjacent = 1 - self.adjacent
        self.neg_neighbors = torch.multinomial(self.neg_adjacent, self.neg_size, replacement=False)

    def sparse_network_reconstruct(self, model, pos_size):

        adj_mask = self.adjacent.clone()
        marker_num = model.generator.marker_num

        current_matrix = torch.stack([model.marker_embeddings for _ in range(model.marker_num)], 0)
        total_matrix = torch.stack([model.marker_embeddings for _ in range(model.marker_num)], 1)
        matrix = torch.cat((current_matrix, total_matrix), 2)
        # print(model.generator.sample_linear_1.weight)
        # print(current_matrix)
        output_matrix = model.generator.sample_linear_1(matrix)
        activated_out_matrix = model.generator.wc(output_matrix)
        final_matrix = model.generator.sample_linear_2(activated_out_matrix).squeeze(2).permute(1, 0)
        output = final_matrix

        for i in range(marker_num):
            for j in range(self.neg_size):
                adj_mask[i][self.neg_neighbors[i][j]] = 1

        masked_output = output * adj_mask
        res = torch.argsort(masked_output, 1, True)
        pred_neighbors = res[:, 0: pos_size]

        result = torch.zeros(marker_num, marker_num)

        for i in range(marker_num):
            for j in range(pos_size):
                result[i][pred_neighbors[i][j]] = 1

        correct = result * self.adjacent

        pred_num = torch.sum(torch.sum(result, 1), 0).item()
        total_num = torch.sum(torch.sum(self.adjacent, 1), 0).item()
        correct_num = torch.sum(torch.sum(correct, 1), 0).item()

        pre = correct_num / pred_num
        rec = correct_num / total_num
        f1 = 2 * pre * rec / (pre + rec)
        return pre, rec, f1

    def sparse_seq_predict(self, model, test_markers, test_times, test_masks, neigh_list, type_of_eval):
        # type_of_eval: 2 / 4 / 6 / 8 : "0" ;    1 / 3 / 5 / 7 / 9 : "1"
        sampled_indices = torch.multinomial(torch.ones(1, test_markers.size()[0]), self.test_size)[0].unsqueeze(-1)
        sampled_marker = test_markers[sampled_indices].squeeze(1)
        sampled_time = test_times[sampled_indices].squeeze(1)
        sampled_mask = test_masks[sampled_indices].squeeze(1)
        res_accu, res_mle = model.generator.test_predict(sampled_marker, sampled_time, sampled_mask,
                                                         neigh_list, model.marker_embeddings, type_of_eval)
        return res_accu, res_mle


class Large_Network_Evaluator(object):

    def __init__(self, marker_size, neg_size, test_size, adj_list):
        self.marker_size = marker_size
        self.neg_size = neg_size
        self.test_size = test_size

        self.adj_list = adj_list
        sample_prob = torch.ones(1, marker_size)
        self.sample_index_list = torch.multinomial(sample_prob, test_size, replacement=False)[0].tolist()
        self.neg_neighbor = []

        out_list = []
        more_list = []

        total_num_out = 0

        for i in range(test_size):
            if len(self.adj_list[self.sample_index_list[i]]) == 0:
                out_list.append(i - total_num_out)
                total_num_out += 1
            elif len(self.adj_list[self.sample_index_list[i]]) >= marker_size - self.neg_size:
                more_list.append(i - total_num_out)

        for i in range(len(out_list)):
            self.sample_index_list.pop(out_list[i])
            self.test_size -= 1
        for i in range(len(self.sample_index_list)):
            neigh_list_i = adj_list[self.sample_index_list[i]]
            if not (i in more_list):
                total_prob = torch.ones(1, marker_size)
                for index in neigh_list_i:
                    total_prob[0][index] = 0.0
                neg_neigh_i = torch.multinomial(total_prob, self.neg_size, replacement=False).tolist()[0]
                self.neg_neighbor.append(neg_neigh_i)
            else:
                potential_neg_list = [_ for _ in range(marker_size)]
                self.neg_neighbor.append(list(set(potential_neg_list)-set(neigh_list_i)))
        self.sample_index = torch.LongTensor(self.sample_index_list)

        self.pos_neighbor = []
        self.eval_mask = torch.zeros(len(self.sample_index_list), self.marker_size)

        for i in range(len(self.sample_index_list)):
            self.pos_neighbor.append(self.adj_list[self.sample_index_list[i]])

        for i in range(len(self.sample_index_list)):
            for pos_item in self.pos_neighbor[i]:
                self.eval_mask[i][pos_item] = 1.0
            for neg_item in self.neg_neighbor[i]:
                self.eval_mask[i][neg_item] = 1.0

    def sparse_network_reconstruct(self, model, pos_size):
        selected_embeddings = model.marker_embeddings[self.sample_index]
        embed_tensor = torch.stack([selected_embeddings for _ in range(self.marker_size)], 0)
        spread_tensor = torch.stack([model.marker_embeddings for _ in range(self.test_size)], 1)
        cat_tensor = torch.cat((spread_tensor, embed_tensor), 2)
        output_matrix = model.generator.sample_linear_1(cat_tensor)
        activated_out_matrix = model.generator.wc(output_matrix)
        final_matrix = model.generator.sample_linear_2(activated_out_matrix).squeeze(2).permute(1, 0)
        output = final_matrix
        output = output * self.eval_mask
        res = torch.argsort(output, 1, True)
        pred_neighbors = res[:, 0: pos_size].tolist()
        total_predict_num = 0
        total_true_num = 0
        correct_num = 0

        for i in range(len(self.sample_index_list)):
            total_predict_num += len(pred_neighbors[i])
            total_true_num += len(self.pos_neighbor[i])
            for j in range(len(pred_neighbors[i])):
                if pred_neighbors[i][j] in self.pos_neighbor[i]:
                    correct_num += 1

        p = correct_num / total_predict_num
        r = correct_num / total_true_num
        f1 = 2 * p * r / (p + r)
        print(p, r, f1)
        return p, r, f1

    def sparse_seq_predict(self, model, test_markers, test_times, test_masks, neigh_list, type_of_eval):
        # type_of_eval: 2 / 4 / 6 / 8 : "0" ;    1 / 3 / 5 / 7 / 9 : "1"
        sampled_indices = torch.multinomial(torch.ones(1, test_markers.size()[0]), self.test_size)[0].unsqueeze(-1)
        sampled_marker = test_markers[sampled_indices].squeeze(1)
        sampled_time = test_times[sampled_indices].squeeze(1)
        sampled_mask = test_masks[sampled_indices].squeeze(1)

        res_accu, res_mle = model.generator.test_predict(sampled_marker, sampled_time, sampled_mask,
                                                         neigh_list, model.marker_embeddings, type_of_eval)
        return res_accu, res_mle


if __name__ == "__main__":
    from model import MainModel
    m = MainModel(10, 5, 2, 10, 20, 10, 10, 10, 4, 5, 0.3, 0, 5, 0.99, 0.001, 0.1)
    a_list = [[],
              [1, 2, 3],
              [1, 2, 3],
              [4, 5, 6],
              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              [1, 2, 3],
              [1, 2, 3, 4, 5, 6, 7, 8],
              [1, 2, 3],
              [1, 2, 3],
              [1, 2, 3]]
    e = Large_Network_Evaluator(10, 4, 3, 6, a_list)
    p, r, f1 = e.sparse_network_reconstruct(m)
    print(p, r, f1)





