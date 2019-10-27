import torch

marker = torch.load("marker.pkl")
time = torch.load("time.pkl")
mask = torch.load("mask.pkl")
adj_list = torch.load("adj_list.pkl")

# Write you own code here to check the data.
# print(marker.size())
print(torch.max(time))
total_num = 0
for i in range(100000):
    total_num += len(adj_list[i])
print(total_num)
print(adj_list[11])
