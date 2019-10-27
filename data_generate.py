import torch
import numpy as np
import random


def sample(return_num, end, num, a1, a2, b1, b2, theta):
    start = max(a1, a2)
    delta = (end - start)/(num - 1)
    candidate = torch.linspace(start+delta, end+delta, num)
    x1 = (candidate-a1) / b1
    x2 = (candidate-a2) / b2
    candi = delta*(theta*(2/(candidate-a1))*x1*x1*np.exp(-x1*x1) + (1-theta)* \
        (2/(candidate-a2))*x2*x2*np.exp(-x2*x2))
    res = candidate[torch.multinomial(candi, return_num, replacement=True)]
    return res


def rand_generate(node_N, cas_num, max_seq_len, prob, a1, a2, b1, b2, theta, num, T):
    adjacency = prob * torch.ones(node_N, node_N, dtype=torch.float)
    for i in range(node_N):
        adjacency[i][i] = 0
    adjacency = torch.bernoulli(adjacency)
    res_mark = torch.zeros(cas_num, max_seq_len, dtype=torch.long)

    start = []
    single_start = [ _ for _ in range(node_N)]

    for i in range(10):
        start = start + single_start

    random.shuffle(start)
    start = torch.LongTensor(start).unsqueeze(0)
    res_mark[:, 0] = start
    res_time = torch.zeros(cas_num, max_seq_len, dtype=torch.float)

    # for i in range(node_N):
    #     adjacency[i][i] = 1

    adj = {}
    adj = {}

    res_mark = torch.zeros(cas_num, max_seq_len, dtype=torch.long)
    print(33)
    start = [_ for _ in range(node_N)]
    random.shuffle(start)
    start = torch.LongTensor(start).unsqueeze(0)

    res_mark[:, 0] = start
    res_time = torch.zeros(cas_num, max_seq_len, dtype=torch.float)

    for i in range(node_N):
        print(i)
        while True:
            print("one")
            neighbors = torch.multinomial(torch.ones(1, node_N).squeeze(0), neigh_num, replacement=False).tolist()
            print(0 in neighbors)
            if not (i in neighbors):
                break

        adj.update({str(i): neighbors})

    adjacent = []
    for i in range(node_N):
        adj.update({str(i):[]})   

    for i in range(node_N):
        for j in range(node_N):
            if adjacency[i][j] == 1:
                adj[str(i)].append(j)

    adj_list = []
    for i in range(node_N):
        adj_list.append(adj[str(i)])

    for i in range(cas_num):
        print(i)
        time = 0
        curr_num = 1

        fathers = np.array([], dtype=float)
        candidates = np.array([], dtype=int)

        results = np.array([int(res_mark[i][0])])
        time_results = np.array([float(res_time[i][0])])
        next_node = int(res_mark[i][0])
        index = -1
        while curr_num < max_seq_len:
            neighbors = np.array(adj[str(int(next_node))])
            print(neighbors)
            candidates = np.append(candidates, neighbors)
            if curr_num == 1:
                fathers = np.append(fathers, np.zeros(neighbors.size))
            else:
                fathers = np.append(fathers, time*np.ones(neighbors.size))
            print(candidates.size)
            index = np.random.choice(candidates.size, 1)[0]
            next_node = candidates[index]
            sample_time = sample(1, T, num, a1, a2, b1, b2, theta)
            time = fathers[index] + sample_time
            time = float(time)
            if time > T:
                break
            results = np.append(results, next_node)
            time_results = np.append(time_results, time)
            
            curr_num += 1
        
        sorted_index = np.argsort(time_results)
    
        time_results = time_results[sorted_index]
        results = results[sorted_index] 
        size = results.size
        
        results = torch.LongTensor(results)
        time_results = torch.FloatTensor(time_results)
        
        res_mark[i][0:size] = results
        res_time[i][0:size] = time_results
        
        times = sample(neighbors.size, T, num, a1, a2, b1, b2, theta).numpy()

        while curr_num < max_seq_len:
            select = np.argmin(times)
            curr_node = neighbors[select]
            res_mark[i][curr_num] = int(curr_node)
            n_time = float(times[select] + res_time[i][curr_num-1])
            if n_time > T:
                break
            res_time[i][curr_num] = n_time
        
            # Now delete the selected one
            neighbors = np.delete(neighbors, select)
            times = np.delete(times, select)
            # Now append the new neighbors
            new_neighbors = np.array(adj[str(curr_node)])
            neighbors = np.append(neighbors, new_neighbors)
            times = np.append(times, sample(new_neighbors.size, T, num, a1, \
                                        a2, b1, b2, theta))
            curr_num += 1

        res_mask = torch.lt(res_time, T).float()
            
    return res_mark, res_time, res_mask, adjacency, adj_list


def rand_large_generate(node_N, cas_num, max_seq_len, neigh_num, a1, a2, b1, b2, theta, num, T):
    adj = {}

    res_mark = torch.zeros(cas_num, max_seq_len, dtype=torch.long)
    start = []
    single_start = [_ for _ in range(node_N)]
    for i in range(10):
        start = start + single_start
    random.shuffle(start)

    start = torch.LongTensor(start).unsqueeze(0)

    res_mark[:, 0] = start
    res_time = torch.zeros(cas_num, max_seq_len, dtype=torch.float)

    for i in range(node_N):
        while True:
            neighbors = torch.multinomial(torch.ones(1, node_N).squeeze(0), neigh_num, replacement=False).tolist()
            if not (i in neighbors):
                break

        adj.update({str(i): neighbors})

    adjacent = []

    for i in range(node_N):
        neigh = adj[str(i)]
        adjacent.append(neigh)

    adj_matrix = torch.zeros(node_N, node_N)

    for i in range(node_N):
        for n in adjacent[i]:
            adj_matrix[i][n] = 1

    for i in range(cas_num):
        time = 0
        curr_num = 1

        fathers = np.array([], dtype=float)
        candidates = np.array([], dtype=int)

        results = np.array([int(res_mark[i][0])])
        time_results = np.array([float(res_time[i][0])])
        next_node = int(res_mark[i][0])
        index = -1
        while curr_num < max_seq_len:
            neighbors = np.array(adj[str(next_node)])
            candidates = np.append(candidates, neighbors)
            if curr_num == 1:
                fathers = np.append(fathers, np.zeros(neighbors.size))
            else:
                fathers = np.append(fathers, time * np.ones(neighbors.size))
            index = np.random.choice(candidates.size, 1)[0]
            next_node = candidates[index]
            sample_time = sample(1, T, num, a1, a2, b1, b2, theta)
            time = fathers[index] + sample_time
            time = float(time)
            if time > T:
                break
            results = np.append(results, next_node)
            time_results = np.append(time_results, time)

            curr_num += 1

        sorted_index = np.argsort(time_results)

        time_results = time_results[sorted_index]
        results = results[sorted_index]
        size = results.size

        results = torch.LongTensor(results)
        time_results = torch.FloatTensor(time_results)

        res_mark[i][0:size] = results
        res_time[i][0:size] = time_results

        times = sample(neighbors.size, T, num, a1, a2, b1, b2, theta).numpy()

        while curr_num < max_seq_len:
            select = np.argmin(times)
            curr_node = neighbors[select]
            res_mark[i][curr_num] = int(curr_node)
            n_time = float(times[select] + res_time[i][curr_num - 1])
            if n_time > T:
                break
            res_time[i][curr_num] = n_time

            # Now delete the selected one
            neighbors = np.delete(neighbors, select)
            times = np.delete(times, select)
            # Now append the new neighbors
            new_neighbors = np.array(adj[str(curr_node)])
            neighbors = np.append(neighbors, new_neighbors)
            times = np.append(times, sample(new_neighbors.size, T, num, a1, \
                                            a2, b1, b2, theta))
            curr_num += 1

        res_mask = torch.lt(res_time, T).float()

    return res_mark, res_time, res_mask, adjacent, adj_matrix


if __name__=="__main__":
    ''' Number of nodes'''
    node_N = 1000
    ''' Number of Cascades'''
    cas_num = 10000
    ''' Max Sequence Length '''
    max_seq_len = 6
    ''' Neighbor Probability'''
    prob = 0.005
    '''Parameters of Rayleigh Distribution'''
    a1 = 0
    a2 = 0
    b1 = 1
    b2 = 1
    theta = 0.5
    num = 10000
    ''' Maximum time constraint, sample from [0,T] '''
    T = 10

    mark_cascade, time_cascade, mask, adj, adj_matrix = rand_large_generate(node_N, cas_num, max_seq_len, \
                                                          5, a1, a2, b1, b2, theta, num, T)

    torch.save(mark_cascade, "marker.pkl")
    torch.save(time_cascade, "time.pkl")
    torch.save(mask, "mask.pkl")
    torch.save(adj, "adj_list.pkl")
    torch.save(adj_matrix, "adj_mat.pkl")


    

    


    
    


        
    


    



