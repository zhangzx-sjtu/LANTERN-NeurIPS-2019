import torch

marker = torch.load("marker.pkl")
time = torch.load("time.pkl")
mask = torch.load("mask.pkl")
adj_mat = torch.load("adj_mat.pkl")
adj_list = torch.load("adj_list.pkl")

# Write you own code here to check the data.
# print(marker.size())
# print(len(adj_list))
total_num = 0
# for i in range(583):
#     total_num += len(adj_list[i])
# print(total_num / 583)
print(torch.max(time))
