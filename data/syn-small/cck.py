import torch

marker = torch.load("marker.pkl")
time = torch.load("time.pkl")
mask = torch.load("mask.pkl")
adj_mat = torch.load("adj_mat.pkl")
adj_list = torch.load("adj_list.pkl")

# Write you own code here to check the dataset.
print(marker.size())

