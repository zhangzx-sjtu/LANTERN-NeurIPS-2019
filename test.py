import torch
from model import MainModel
from evaluation import Small_Network_Evaluator, Large_Network_Evaluator
from math import ceil

def list_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


def list_sub(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] - b[i])
    return c


def list_mul(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] * b[i])
    return c


def list_div(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] / b[i])
    return c


def cal_mean(l):
    # l is a 2-dim list [[], [], []] .. etc
    list_len = len(l)
    each_list_len = len(l[0])
    sum_list = l[0]
    for i in range(list_len - 1):
        sum_list = list_add(sum_list, l[i + 1])
    num = [list_len for _ in range(each_list_len)]
    mean = list_div(sum_list, num)
    return mean


def cal_var(l):
    # l is a 2-dim list [[], [], []] .. etc
    list_len = len(l)
    each_list_len = len(l[0])
    mean = cal_mean(l)
    final_list = list_mul(list_sub(l[0], mean), list_sub(l[0], mean))
    for i in range(list_len - 1):
        final_list = list_add(final_list, list_mul(list_sub(l[i + 1], mean), list_sub(l[i + 1], mean)))
    num = [list_len for _ in range(each_list_len)]
    var = list_div(final_list, num)
    return var

dataset = "memetracker"
small = 0 # 1 for small and 0 for large
type_of_eval = 0
marker_num = 583
cascade_num = 6700
pos_num = [25, 30, 35]
neg_num = 5
test_size = 100
d_model = 8
max_time = 8
head = 2
candi_size = 583

train_data_num = ceil(cascade_num / 10 * 9)
syn_small_marker = torch.load("data/" + dataset + "/marker.pkl").cuda(0)[train_data_num:, :]
syn_small_time = torch.load("data/" + dataset + "/time.pkl").cuda(0)[train_data_num:, :]
syn_small_mask = torch.load("data/" + dataset + "/mask.pkl").cuda(0)[train_data_num:, :]
if small:
    syn_small_adj_mat = torch.load("data/" + dataset + "/adj_mat.pkl")
syn_small_adj_list = torch.load("data/" + dataset + "/adj_list.pkl")

syn_small_main = MainModel(marker_num, 5, d_model, d_model, 2 * d_model, d_model, d_model, d_model, head, candi_size, max_time, 0.3, 0, 10, 0.99, 0.001, 0.1)
syn_small_main.load_state_dict(torch.load("model/main-" + dataset + "-512.pt"))

syn_small_main.cpu().eval()
if small:
    syn_small_evaluator = Small_Network_Evaluator(marker_num, neg_num, test_size, syn_small_adj_mat)
else:
    syn_small_evaluator = Large_Network_Evaluator(marker_num, neg_num, test_size, syn_small_adj_list)

main= []
main_mean= []

for i in range(3):
    main_i= []
    pos_size = pos_num[i]
    for i in range(5):
        if small:
            syn_small_evaluator = Small_Network_Evaluator(marker_num, neg_num, test_size, syn_small_adj_mat)
        else:
            syn_small_evaluator = Large_Network_Evaluator(marker_num, neg_num, test_size, syn_small_adj_list)
        p, r, f1 = syn_small_evaluator.sparse_network_reconstruct(syn_small_main, pos_size)
        main_i.append([p, r, f1])

    main.append(main_i)

for i in range(3):
    main_mean.append(cal_mean(main[i]))

for i in range(3):
    print("main_mean--K"+str(i+1), main_mean[i])
