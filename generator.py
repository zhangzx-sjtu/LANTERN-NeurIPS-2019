import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer


class Generator(nn.Module):

    def __init__(self, marker_num, neighbor_num, embed_dim, d_model, d_inner, d_q, d_k,
                 d_v, n_head, candi_size, max_time, beta, cuda_id, dropout=0.1):

        super(Generator, self).__init__()
        # Modules
        self.encoder = Transformer(n_head, d_model, d_inner, d_q, d_k, d_v, cuda_id, dropout)
        self.time_embed_linear = nn.Linear(1, embed_dim)
        self.embed_linear = nn.Linear(embed_dim, d_model)
        self.embed_ac = nn.LeakyReLU()
        self.sample_linear_1 = nn.Linear(2 * embed_dim, embed_dim)
        self.wc = nn.LeakyReLU()
        self.sample_linear_2 = nn.Linear(embed_dim, 1)
        self.marker_linear = nn.Linear(embed_dim + d_model, 1)
        self.time_linear = nn.Linear(d_model, 1)
        # Constants
        self.max_time = max_time
        self.d_model = d_model
        self.cuda_id = cuda_id
        self.beta = beta
        self.embed_dim = embed_dim
        self.marker_num = marker_num
        self.sample_size = neighbor_num
        self.candi_size = candi_size
        # Descendants in an epoch
        self.neighbor_list = None
        self.neighbor_prob = None
        self.candidates = None

    def sample_neighbors(self, embedding):
        # Initialize descendants at the beginning of an epoch
        self.candidates = torch.multinomial(torch.ones(1, self.marker_num).cuda(self.cuda_id), self.candi_size)
        total_candidates = torch.stack([self.candidates[0] for _ in range(self.marker_num)], 0)
        sel_candi_embeds = embedding[self.candidates][0]
        current_matrix = torch.stack([sel_candi_embeds for _ in range(self.marker_num)], 0)
        total_matrix = torch.stack([embedding for _ in range(self.candi_size)], 1)
        matrix = torch.cat((current_matrix, total_matrix), 2)

        output_matrix = self.sample_linear_1(matrix)
        activated_out_matrix = self.wc(output_matrix)
        final_matrix = self.sample_linear_2(activated_out_matrix).squeeze(2)
        prob_res = torch.softmax(final_matrix, 1)
        # Initialize descendants and probabilities
        print(prob_res.size())
        print(self.sample_size)
        neighbors = torch.multinomial(prob_res, self.sample_size)
        p_list = torch.gather(prob_res, 1, neighbors)
        self.neighbor_list = torch.gather(total_candidates, 1, neighbors)
        self.neighbor_prob = p_list

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

    def forward(self, marker_data, time_data, mask_data, embedding):
        # Forward Propagation
        d_size = time_data.size()
        # Initialize data
        self.index = 0
        marker_res = marker_data[:, 0:1].clone()
        time_res = time_data[:, 0:1].clone()
        mask_res = mask_data[:, 0:1].clone()
        data_input = torch.zeros(d_size[0], 0, self.d_model).cuda(self.cuda_id)
        candidate_list = marker_data[:, 0:1]
        prob_list = torch.ones(d_size[0], 1).cuda(self.cuda_id)
        chosen_index = torch.zeros(d_size[0], 1, dtype=torch.long).cuda(self.cuda_id)
        neighbor_prob_record = torch.ones(d_size[0], 1).cuda(self.cuda_id)
        total_neighbor_prob = torch.ones(d_size[0], 1).cuda(self.cuda_id)
        total_sample_prob = torch.ones(d_size[0], 1).cuda(self.cuda_id)

        # Generating Cascades
        while self.index < d_size[1] - 1:
            last_marker = marker_res[:, self.index: self.index + 1]
            last_time = time_res[:, self.index: self.index + 1]
            new_vector = self.get_embedding(last_marker, last_time, embedding)
            model_dim_vector = self.embed_ac(self.embed_linear(new_vector))
            data_input = torch.cat((data_input, model_dim_vector), 1)
            intensity = self.encoder.forward(data_input, self.index)
            # Time Decoding
            new_time = last_time + F.softplus(self.time_linear(intensity.squeeze(1)))
            # Causal Descendants
            time_res = torch.cat((time_res, new_time), 1)
            new_mask = torch.lt(new_time, self.max_time).float()
            neighbor = self.neighbor_list[last_marker].squeeze(1)
            prob_neighbor = self.neighbor_prob[last_marker].squeeze(1)
            neighbor_prob_record = torch.cat((neighbor_prob_record, prob_neighbor), 1)
            # Intensity Function
            neighbor_inf = embedding[neighbor]
            intensity_inf = torch.stack([(intensity.squeeze(1)) for _ in range(self.sample_size)], 1)
            inf_matrix = torch.cat((neighbor_inf, intensity_inf), 2)
            # Marker Decoding
            marker_weight = self.marker_linear(inf_matrix).squeeze(2)
            marker_prob = torch.softmax(marker_weight, 1)
            candidate_list = torch.cat((candidate_list, neighbor[:, 1:]), 1)
            chosen_prob = torch.gather(prob_list, 1, chosen_index)
            attached_prob = chosen_prob * marker_prob
            for i in range(d_size[0]):
                prob_list[i][chosen_index[i]] = attached_prob[i][0]
            prob_list = torch.cat((prob_list, attached_prob[:, 1:]), 1)
            chosen_index = torch.multinomial(prob_list, 1)
            new_markers = torch.gather(candidate_list, 1, chosen_index)
            # Record Probabilities for BP
            selected_neighbor_prob = torch.gather(neighbor_prob_record, 1, chosen_index)
            total_neighbor_prob = torch.cat((total_neighbor_prob, selected_neighbor_prob), 1)
            selected_sample_prob = torch.gather(prob_list, 1, chosen_index)
            total_sample_prob = torch.cat((total_sample_prob, selected_sample_prob), 1)
            self.index += 1
            # Mark down the Results
            marker_res = torch.cat((marker_res, new_markers), 1)
            mask_res = torch.cat((mask_res, new_mask), 1)

        return marker_res, time_res, mask_res, total_neighbor_prob, total_sample_prob

    def test_predict(self, test_marker, test_time, test_mask, true_neigh_list, embedding, type_eval):
        size = test_time.size()
        if type_eval:
            time_mse = [0, 0, 0, 0, 0]
            marker_correct_count = [0, 0, 0, 0, 0]
            marker_ttl_count = [0, 0, 0, 0, 0]
        else:
            time_mse = [0, 0, 0, 0]
            marker_correct_count = [0, 0, 0, 0]
            marker_ttl_count = [0, 0, 0, 0]

        # For each cascade
        for j in range(size[0]):
            # Get single marker, time and mask (1, 6)
            single_marker = test_marker[j: j + 1, :]
            single_time = test_time[j: j + 1, :]
            single_mask = test_mask[j: j + 1, :]

            if type_eval:
                if torch.sum(single_mask, 1).item() < 6:
                    continue
            else:
                if torch.sum(single_mask, 1).item() < 5:
                    continue

            length = torch.sum(single_mask, 1).item()

            for k in range(len(marker_ttl_count)):
                marker_ttl_count[k] += 1

            sample_prob = torch.ones(1, 1).cuda(self.cuda_id)
            candidates = single_marker[:, 0:1]

            total_candidates = []
            total_probabilities = []

            data_input = torch.zeros(1, 0, self.d_model).cuda(self.cuda_id)
            last_index = 0

            # First store previous informations
            for i in range(int(length)):
                total_candidates.append(candidates.clone())
                total_probabilities.append(sample_prob.clone())

                new_vector = self.get_embedding(single_marker[:, i:i+1], single_time[:, i:i+1], embedding)
                data_input = torch.cat((data_input, new_vector), 1)
                intensity = self.encoder.forward(data_input, i)

                if true_neigh_list[single_marker[0][i].item()] == []:
                    empiri_list = [single_marker[0][i].item()]
                else:
                    empiri_list = true_neigh_list[single_marker[0][i].item()]

                candidate_i = torch.LongTensor([empiri_list]).cuda(self.cuda_id)
                neigh_size = len(empiri_list)
                neighbor_inf = embedding[candidate_i]
                intensity_inf = torch.stack([(intensity.squeeze(1)) for _ in range(neigh_size)], 1)
                inf_matrix = torch.cat((neighbor_inf, intensity_inf), 2)
                marker_weight = self.marker_linear(inf_matrix).squeeze(2)
                marker_prob = torch.softmax(marker_weight, 1)

                if last_index != -1:
                    attach_prob = sample_prob[0][last_index].clone()
                    sample_prob[0][last_index] = sample_prob[0][last_index] * marker_prob[0][0]
                    candidates[0][last_index] = candidate_i[0][0]

                    new_marker_prob = attach_prob * marker_prob[:, 1:]
                    sample_prob = torch.cat((sample_prob, new_marker_prob), 1)
                    candidates = torch.cat((candidates, candidate_i[:, 1:]), 1)

                else:
                    sample_prob = torch.cat((sample_prob, marker_prob), 1)
                    candidates = torch.cat((candidates, candidate_i), 1)

                if i != length - 1:
                    if single_marker[0][i+1].item() in candidates.tolist()[0]:
                        last_index = torch.argmax(candidates == single_marker[0][i+1].item()).item()
                    else:
                        last_index = -1

            for p in range(len(marker_ttl_count)):
                curr_id = int((length - 1) / len(marker_ttl_count) * p)

                new_vector = self.get_embedding(single_marker[:, curr_id:curr_id + 1],
                                                single_time[:, curr_id:curr_id + 1], embedding)
                data_input = torch.cat((data_input, new_vector), 1)
                intensity = self.encoder.forward(data_input, curr_id)

                # Calculate MSE
                delta_time = F.softplus(self.time_linear(intensity.squeeze(1)))
                time_bias = ((delta_time - (single_time[:, curr_id + 1:curr_id + 2] - single_time[:, curr_id :curr_id + 1])) *
                             (delta_time - (single_time[:, curr_id + 1:curr_id + 2] - single_time[:, curr_id :curr_id + 1]))).item()
                time_mse[p] += time_bias

                # Calculate accu for markers
                new_vector = self.get_embedding(single_marker[:, curr_id + 1:curr_id + 2],
                                                single_time[:, curr_id + 1:curr_id + 2], embedding)
                data_input = torch.cat((data_input, new_vector), 1)
                intensity = self.encoder.forward(data_input, curr_id + 1)
                curr_candi = total_candidates[curr_id + 1]
                curr_prob = total_probabilities[curr_id + 1]
                samp_neighs = self.neighbor_list[single_marker[:, curr_id + 1: curr_id + 2]].squeeze(1)
                neighbor_inf = embedding[samp_neighs]
                intensity_inf = torch.stack([(intensity.squeeze(1)) for _ in range(self.sample_size)], 1)
                inf_matrix = torch.cat((neighbor_inf, intensity_inf), 2)
                marker_weight = self.marker_linear(inf_matrix).squeeze(2)
                marker_prob = torch.softmax(marker_weight, 1)

                if single_marker[0, curr_id + 1] in curr_candi.tolist()[0]:
                    curr_index = torch.argmax(curr_candi == single_marker[0][curr_id + 1].item()).item()
                else:
                    curr_index = -1

                if curr_index != -1:
                    attach_prob = curr_prob[0][curr_index]
                    curr_candi[0][curr_index] = samp_neighs[0][0]
                    curr_prob[0][curr_index] = attach_prob * marker_prob[0][0]
                    next_candi = torch.cat((curr_candi, samp_neighs[:, 1:]), 1)
                    next_prob = torch.cat((curr_prob, marker_prob[:, 1:]), 1)
                else:
                    next_candi = torch.cat((curr_candi, samp_neighs), 1)
                    next_prob = torch.cat((curr_prob, marker_prob), 1)
                predict = next_candi[0][torch.multinomial(next_prob, 1)[0][0].item()].item()
                if predict in total_candidates[curr_id + 1][0].tolist():
                    marker_correct_count[p] += 1

        accu = list(map(lambda x: x[0] / x[1], zip(marker_correct_count, marker_ttl_count)))
        mse = list(map(lambda x: x[0] / x[1], zip(time_mse, marker_ttl_count)))

        return accu, mse


class RNN_Generator(nn.Module):

    def __init__(self, marker_num, neighbor_num, embed_dim, d_model, candi_size, max_time, beta, cuda_id, dropout=0.1):
        super(RNN_Generator, self).__init__()
        # Modules
        self.encoder = nn.RNN(d_model, d_model, 1, batch_first=True, dropout=dropout)
        self.time_embed_linear = nn.Linear(1, embed_dim)
        self.embed_linear = nn.Linear(embed_dim, d_model)
        self.embed_ac = nn.LeakyReLU()
        self.sample_linear_1 = nn.Linear(2 * embed_dim, embed_dim)
        self.wc = nn.LeakyReLU()
        self.sample_linear_2 = nn.Linear(embed_dim, 1)
        self.marker_linear = nn.Linear(embed_dim + d_model, 1)
        self.time_linear = nn.Linear(d_model, 1)
        # Constants
        self.embed_dim = embed_dim
        self.max_time = max_time
        self.d_model = d_model
        self.cuda_id = cuda_id
        self.index = 0
        self.beta = beta
        self.marker_num = marker_num
        self.sample_size = neighbor_num
        self.candi_size = candi_size
        # Initialize Neighbor List in One Epoch
        self.neighbor_list = None
        self.neighbor_prob = None
        self.candidates = None

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

    def sample_neighbors(self, embedding):
        # Initialize descendants at the beginning of an epoch
        self.candidates = torch.multinomial(torch.ones(1, self.marker_num).cuda(self.cuda_id), self.candi_size)
        total_candidates = torch.stack([self.candidates[0] for _ in range(self.marker_num)], 0)
        sel_candi_embeds = embedding[self.candidates][0]
        current_matrix = torch.stack([sel_candi_embeds for _ in range(self.marker_num)], 0)
        total_matrix = torch.stack([embedding for _ in range(self.candi_size)], 1)
        matrix = torch.cat((current_matrix, total_matrix), 2)
        output_matrix = self.sample_linear_1(matrix)
        activated_out_matrix = self.wc(output_matrix)
        final_matrix = self.sample_linear_2(activated_out_matrix).squeeze(2)
        prob_res = torch.softmax(final_matrix, 1)
        # Initialize descendants and probabilities
        neighbors = torch.multinomial(prob_res, self.sample_size)
        p_list = torch.gather(prob_res, 1, neighbors)
        self.neighbor_list = torch.gather(total_candidates, 1, neighbors)
        self.neighbor_prob = p_list

    def forward(self, marker_data, time_data, mask_data, embedding):
        size = time_data.size()
        self.index = 0
        # Initialize Data Inputs and Outputs
        marker_res = torch.zeros(size, dtype=torch.long).cuda(self.cuda_id)
        time_res = time_data[:, 0:1]
        mask_res = torch.zeros(size, dtype=torch.float).cuda(self.cuda_id)
        marker_res[:, 0:1] = marker_data[:, 0:1]
        mask_res[:, 0:1] = mask_data[:, 0:1]
        data_input = torch.zeros(size[0], 0, self.d_model).cuda(self.cuda_id)
        intense_res = torch.zeros(size[0], 0, self.d_model).cuda(self.cuda_id)
        candidate_list = marker_data[:, 0:1]
        prob_list = torch.ones(size[0], 1).cuda(self.cuda_id)
        chosen_index = torch.zeros(size[0], 1, dtype=torch.long).cuda(self.cuda_id)
        # Hidden States of RNN Encoder
        hidden_state = torch.zeros(1, size[0], self.d_model).cuda(self.cuda_id)
        # Output Probabilities for Expectation Maximization
        neighbor_prob_record = torch.ones(size[0], 1).cuda(self.cuda_id)
        total_neighbor_prob = torch.ones(size[0], 1).cuda(self.cuda_id)
        total_sample_prob = torch.ones(size[0], 1).cuda(self.cuda_id)
        # Sequence Generation
        while self.index < size[1] - 1:
            last_marker = marker_res[:, self.index: self.index + 1]
            last_time = time_res[:, self.index: self.index + 1]
            new_vector = self.get_embedding(last_marker, last_time, embedding)
            model_dim_vector = self.embed_ac(self.embed_linear(new_vector))
            data_input = torch.cat((data_input, model_dim_vector), 1)
            rnn_output = self.encoder(data_input[:, self.index: self.index + 1, :], hidden_state)

            intensity = rnn_output[0]
            hidden_state = rnn_output[1]
            # Time Decoding
            new_time = last_time + F.softplus(self.time_linear(intensity.squeeze(1)))
            time_res = torch.cat((time_res, new_time), 1)

            new_mask = torch.lt(new_time, self.max_time).long()
            intense_res = torch.cat((intense_res, intensity), 1)
            neighbor = self.neighbor_list[last_marker].squeeze(1)
            prob_neighbor = self.neighbor_prob[last_marker].squeeze(1)
            # Intensity Function Stacks
            neighbor_prob_record = torch.cat((neighbor_prob_record, prob_neighbor), 1)
            neighbor_inf = embedding[neighbor]
            intensity_inf = torch.stack([(intensity.squeeze(1)) for _ in range(self.sample_size)], 1)
            inf_matrix = torch.cat((neighbor_inf, intensity_inf), 2)
            # Marker Decoding
            marker_weight = self.marker_linear(inf_matrix).squeeze(2)
            marker_prob = torch.softmax(marker_weight, 1)
            # Marker Sampling
            candidate_list = torch.cat((candidate_list, neighbor[:, 1:]), 1)
            chosen_prob = torch.gather(prob_list, 1, chosen_index)
            attached_prob = chosen_prob * marker_prob
            for i in range(size[0]):
                prob_list[i][chosen_index[i]] = attached_prob[i][0]
            prob_list = torch.cat((prob_list, attached_prob[:, 1:]), 1)
            chosen_index = torch.multinomial(prob_list, 1)

            new_markers = torch.gather(candidate_list, 1, chosen_index)
            # Record the marker probabilities for optimization
            selected_neighbor_prob = torch.gather(neighbor_prob_record, 1, chosen_index)
            total_neighbor_prob = torch.cat((total_neighbor_prob, selected_neighbor_prob), 1)
            selected_sample_prob = torch.gather(prob_list, 1, chosen_index)
            total_sample_prob = torch.cat((total_sample_prob, selected_sample_prob), 1)

            self.index += 1
            marker_res = torch.cat((marker_res, new_markers), 1)
            mask_res[:, self.index: self.index + 1] = new_mask

        return marker_res, time_res, mask_res, total_neighbor_prob, total_sample_prob

    def test_predict(self, test_marker, test_time, test_mask, true_neigh_list, embedding, type_eval):
        size = test_time.size()
        if type_eval:
            time_mse = [0, 0, 0, 0, 0]
            marker_correct_count = [0, 0, 0, 0, 0]
            marker_ttl_count = [0, 0, 0, 0, 0]
        else:
            time_mse = [0, 0, 0, 0]
            marker_correct_count = [0, 0, 0, 0]
            marker_ttl_count = [0, 0, 0, 0]

        # For each cascade
        for j in range(size[0]):
            # Get single marker, time and mask (1, 6)
            single_marker = test_marker[j: j + 1, :]
            single_time = test_time[j: j + 1, :]
            single_mask = test_mask[j: j + 1, :]

            if type_eval:
                if torch.sum(single_mask, 1).item() < 6:
                    continue
            else:
                if torch.sum(single_mask, 1).item() < 5:
                    continue

            length = torch.sum(single_mask, 1).item()
            for k in range(len(marker_ttl_count)):
                marker_ttl_count[k] += 1

            sample_prob = torch.ones(1, 1).cuda(self.cuda_id)
            candidates = single_marker[:, 0:1]

            total_candidates = []
            total_probabilities = []

            hidden_state = torch.zeros(1, 1, self.d_model).cuda(self.cuda_id)
            total_hidden_state = torch.zeros(1, 1, self.d_model).cuda(self.cuda_id)

            data_input = torch.zeros(1, 0, self.d_model).cuda(self.cuda_id)
            last_index = 0

            # First store previous information
            for i in range(int(length)):
                total_candidates.append(candidates.clone())
                total_probabilities.append(sample_prob.clone())
                new_vector = self.get_embedding(single_marker[:, i:i+1], single_time[:, i:i+1], embedding)
                data_input = torch.cat((data_input, new_vector), 1)
                rnn_output = self.encoder(data_input[:, i: i+1, :], hidden_state)

                intensity = rnn_output[0]
                hidden_state = rnn_output[1]
                total_hidden_state = torch.cat((total_hidden_state, hidden_state.clone()), 1)

                if true_neigh_list[single_marker[0][i].item()] == []:
                    empiri_list = [single_marker[0][i].item()]
                else:
                    empiri_list = true_neigh_list[single_marker[0][i].item()]

                candidate_i = torch.LongTensor([empiri_list]).cuda(self.cuda_id)
                neigh_size = len(empiri_list)

                neighbor_inf = embedding[candidate_i]
                intensity_inf = torch.stack([(intensity.squeeze(1)) for _ in range(neigh_size)], 1)
                inf_matrix = torch.cat((neighbor_inf, intensity_inf), 2)
                marker_weight = self.marker_linear(inf_matrix).squeeze(2)
                marker_prob = torch.softmax(marker_weight, 1)

                if last_index != -1:
                    attach_prob = sample_prob[0][last_index].clone()
                    sample_prob[0][last_index] = sample_prob[0][last_index] * marker_prob[0][0]
                    candidates[0][last_index] = candidate_i[0][0]

                    new_marker_prob = attach_prob * marker_prob[:, 1:]
                    sample_prob = torch.cat((sample_prob, new_marker_prob), 1)
                    candidates = torch.cat((candidates, candidate_i[:, 1:]), 1)

                else:
                    sample_prob = torch.cat((sample_prob, marker_prob), 1)
                    candidates = torch.cat((candidates, candidate_i), 1)

                if i != length - 1:
                    if single_marker[0][i+1].item() in candidates.tolist()[0]:
                        last_index = torch.argmax(candidates == single_marker[0][i+1].item()).item()
                    else:
                        last_index = -1

            for p in range(len(marker_ttl_count)):
                curr_id = int((length - 1) / len(marker_ttl_count) * p)

                new_vector = self.get_embedding(single_marker[:, curr_id:curr_id + 1],
                                                single_time[:, curr_id:curr_id + 1], embedding)

                data_input = torch.cat((data_input, new_vector), 1)
                rnn_output = self.encoder(data_input[:, curr_id: curr_id + 1, :], total_hidden_state[:, curr_id: curr_id + 1, :])

                intensity = rnn_output[0]

                delta_time = single_time[:, curr_id:curr_id + 1] + F.softplus(self.time_linear(intensity.squeeze(1)))
                time_bias = ((delta_time - (single_time[:, curr_id + 1:curr_id + 2] - single_time[:, curr_id:curr_id + 1])) *
                             (delta_time - (single_time[:, curr_id + 1:curr_id + 2] - single_time[:, curr_id:curr_id + 1]))).item()
                time_mse[p] += time_bias

                # Calculate accu for markers
                new_vector = self.get_embedding(single_marker[:, curr_id + 1:curr_id + 2],
                                                single_time[:, curr_id + 1:curr_id + 2], embedding)
                data_input = torch.cat((data_input, new_vector), 1)
                rnn_output = self.encoder(data_input[:, curr_id + 1: curr_id + 2, :], total_hidden_state[:, curr_id + 1: curr_id + 2, :])

                intensity = rnn_output[0]

                curr_candi = total_candidates[curr_id + 1]
                curr_prob = total_probabilities[curr_id + 1]
                samp_neighs = self.neighbor_list[single_marker[:, curr_id + 1: curr_id + 2]].squeeze(1)
                neighbor_inf = embedding[samp_neighs]
                intensity_inf = torch.stack([(intensity.squeeze(1)) for _ in range(self.sample_size)], 1)
                inf_matrix = torch.cat((neighbor_inf, intensity_inf), 2)
                marker_weight = self.marker_linear(inf_matrix).squeeze(2)
                marker_prob = torch.softmax(marker_weight, 1)

                if single_marker[0, curr_id + 1] in curr_candi.tolist()[0]:
                    curr_index = torch.argmax(curr_candi == single_marker[0][curr_id + 1].item()).item()
                else:
                    curr_index = -1

                if curr_index != -1:
                    attach_prob = curr_prob[0][curr_index]
                    curr_candi[0][curr_index] = samp_neighs[0][0]
                    curr_prob[0][curr_index] = attach_prob * marker_prob[0][0]
                    next_candi = torch.cat((curr_candi, samp_neighs[:, 1:]), 1)
                    next_prob = torch.cat((curr_prob, marker_prob[:, 1:]), 1)
                else:
                    next_candi = torch.cat((curr_candi, samp_neighs), 1)
                    next_prob = torch.cat((curr_prob, marker_prob), 1)

                predict = next_candi[0][torch.multinomial(next_prob, 1)[0][0].item()].item()

                if predict in total_candidates[curr_id + 1][0].tolist():
                    marker_correct_count[p] += 1

        accu = list(map(lambda x: x[0] / x[1], zip(marker_correct_count, marker_ttl_count)))
        mse = list(map(lambda x: x[0] / x[1], zip(time_mse, marker_ttl_count)))

        return accu, mse


if __name__ == "__main__":
    g = Generator(7, 2, 5, 6, 10, 5, 5, 5, 2, 4, 1, 0.3, 0, 0.1).cuda(0)
