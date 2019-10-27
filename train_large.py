import torch
import argparse
import logging

from torch.utils.data import TensorDataset, DataLoader
from model import MainModel, PR_Model, RNN_Model
from evaluation import Large_Network_Evaluator
from math import ceil

parser = argparse.ArgumentParser()

# device (compulsory)
parser.add_argument('-cuda', default=0, type=int)
# data (compulsory)
parser.add_argument('-data', default='syn-large', type=str)
parser.add_argument('-d_marker', default=100000, type=int)
parser.add_argument('-d_cascade', default=100000, type=int)
# model
parser.add_argument('-d_model', default=16, type=int)
parser.add_argument('-neighbor', default=5, type=int)
parser.add_argument('-d_inner', default=32, type=int)
parser.add_argument('-d_head', default=1, type=int)
parser.add_argument('-max_time', default=8, type=float)
parser.add_argument('-embed_ratio', default=0.3, type=int)
parser.add_argument('-sample', default=1, type=int)
parser.add_argument("-model", default="main", type=str)
# optimize (compulsory)
parser.add_argument('-epoches', default=10000, type=int)
parser.add_argument('-batch', default=512, type=int)
parser.add_argument('-d_update', default=1, type=int)
parser.add_argument('-g_update', default=1, type=int)
parser.add_argument('-model_dir', default="main", type=str)
parser.add_argument('-log_dir', default="main", type=str)
# optimize
parser.add_argument('-discount', default=0.99, type=float)
parser.add_argument('-regular', default=0.001, type=float)
parser.add_argument('-dropout', default=0.1, type=float)
parser.add_argument('-lr', default=1e-3, type=float)
parser.add_argument('-lr_g', default=1e-4, type=float)
parser.add_argument('-lr_d', default=1e-4, type=float)
parser.add_argument('-seed', default=111, type=int)
# evaluation(compulsory)
# questionable
parser.add_argument('-neg_size', default=5, type=int)
parser.add_argument('-pos_size', default=[30, 35, 40], type=list)
parser.add_argument('-test_sample', default=100, type=int)

args = parser.parse_args()
log_dir = args.model + '-' + args.data + '-' + str(args.batch)
model_dir = args.model + '-' + args.data + '-' + str(args.batch)
logging.basicConfig(filename="log/" + log_dir + ".log", level=logging.DEBUG)

torch.manual_seed(args.seed)

data_dir = "data/" + args.data + "/"

if args.model == "main":
    network = MainModel(args.d_marker, args.neighbor, args.d_model, args.d_inner, args.d_model,
                        args.d_model, args.d_model, args.d_head, args.max_time, args.embed_ratio,
                        args.cuda, args.sample, args.discount, args.regular, dropout=args.dropout).cuda(args.cuda)
if args.model == "pr":
    network = PR_Model(args.d_marker, args.neighbor, args.d_model, args.d_inner, args.d_model,
                       args.d_model, args.d_model, args.d_head, args.max_time, args.embed_ratio,
                       args.cuda, args.sample, args.discount, args.regular, dropout=args.dropout).cuda(args.cuda)

if args.model == "rnn":
    network = RNN_Model(args.d_marker, args.neighbor, args.d_model, args.max_time, args.embed_ratio, args.cuda,
                        args.sample, args.discount, args.regular, dropout=args.dropout).cuda(args.cuda)

network.generator.sample_linear.cpu()

marker_data = torch.load(data_dir + "marker.pkl").cuda(args.cuda)
time_data = torch.load(data_dir + "time.pkl").cuda(args.cuda)
mask_data = torch.load(data_dir + "mask.pkl").cuda(args.cuda)
adj_list = torch.load(data_dir + "adj_list.pkl")

train_data_num = int(args.d_cascade / 10 * 9)

train_marker = marker_data[0: train_data_num, :]
train_time = time_data[0: train_data_num, :]
train_mask = mask_data[0: train_data_num, :]

test_marker = marker_data[train_data_num:, :]
test_time = time_data[train_data_num:, :]
test_mask = mask_data[train_data_num:, :]

d_set = TensorDataset(train_marker, train_time, train_mask)
d_lder = DataLoader(d_set, shuffle=True, batch_size=args.batch)

evaluator = Large_Network_Evaluator(args.d_marker, args.neg_size, args.test_sample, adj_list, 1)

# Training and Testing
max_p, max_r, max_f1 = 0, 0, 0
# Inverse Reinforcement Learning
g_optimizer = torch.optim.Adam([{"params": network.marker_embeddings, "lr": args.lr_g},
                                {"params": network.generator.parameters(), "lr": args.lr_g}], lr=args.lr_g, betas=[0.9, 0.99])
if args.model != "pr":
    d_optimizer = torch.optim.Adam([{"params": network.discriminator.parameters(), "lr": args.lr_d}],
                                   lr=args.lr_d, betas=[0.9, 0.99])

# Training and Testing
for j in range(args.epoches):

    print("----------------------------------------------")
    print("Epoch: " + str(j + 1) + "/" + str(args.epoches))
    logging.info("Epoch: " + str(j + 1) + "/" + str(args.epoches))

    d_avg_loss = 0
    g_avg_loss = 0

    i = 0

    if args.model != "pr":
        for i in range(args.d_update):
            print("discriminator update: " + str(i + 1) + "/" + str(args.d_update))
            logging.info("discriminator update: " + str(i + 1) + "/" + str(args.d_update))
            batch_id = 0
            batch_num = ceil(train_data_num / args.batch)
            d_avg_loss = 0

            for batch_marker, batch_time, batch_mask in d_lder:
                print("Batch ID: " + str(batch_id + 1) + "/" + str(batch_num))
                loss_d, loss_g = network.forward(batch_marker, batch_time, batch_mask, 1)
                d_optimizer.zero_grad()
                loss_d.backward()
                d_optimizer.step()
                d_avg_loss += loss_d.item()
                batch_id += 1
                if batch_id > 30:
                    break

            print("d_loss: " + str(d_avg_loss))
            logging.info("d_loss: " + str(d_avg_loss))

    for i in range(args.g_update):
        print("generator update: " + str(i + 1) + "/" + str(args.g_update))
        logging.info("generator update: " + str(i + 1) + "/" + str(args.g_update))
        batch_id = 0
        batch_num = ceil(train_data_num / args.batch)
        g_avg_loss = 0

        for batch_marker, batch_time, batch_mask in d_lder:
            if args.model != "pr":
                loss_d, loss_g = network.forward(batch_marker, batch_time, batch_mask, 1)
            else:
                loss_g = network.forward(batch_marker, batch_time, batch_mask, 0)
            print("Batch ID: " + str(batch_id + 1) + "/" + str(batch_num))
            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()
            g_avg_loss += loss_g.item()
            batch_id += 1
            if batch_id > 30:
                break

        print("g_loss: " + str(g_avg_loss))
        logging.info("g_loss: " + str(g_avg_loss))

        # Testing
        network.cpu().eval()
        p_1, r_1, f1_1 = evaluator.sparse_network_reconstruct(network, args.pos_size[0])
        p_2, r_2, f1_2 = evaluator.sparse_network_reconstruct(network, args.pos_size[1])
        p_3, r_3, f1_3 = evaluator.sparse_network_reconstruct(network, args.pos_size[2])
        print(p_3, r_3, f1_3)
        if f1_3 > max_f1:
            max_f1 = f1_3
            max_p = p_3
            max_r = r_3
            torch.save(network.state_dict(), "model/" + model_dir + ".pt")
        network.cuda(args.cuda)
        network.generator.sample_linear.cpu()
        print("Network Reconstruction Results (K=" + str(args.pos_size[0]) + "): ")
        print("Prec: " + str(p_1) + " Rec: " + str(r_1) + " F1: " + str(f1_1))
        print("Network Reconstruction Results (K=" + str(args.pos_size[1]) + "): ")
        print("Prec: " + str(p_2) + " Rec: " + str(r_2) + " F1: " + str(f1_2))
        print("Network Reconstruction Results (K=" + str(args.pos_size[2]) + "): ")
        print("Prec: " + str(max_p) + " Rec: " + str(max_r) + " F1: " + str(max_f1))
        logging.info("K=" + str(args.pos_size[0]) + "Prec: " + str(p_1) + " Rec: " + str(r_1) + " F1: " + str(f1_1))
        logging.info("K=" + str(args.pos_size[1]) + "Prec: " + str(p_2) + " Rec: " + str(r_2) + " F1: " + str(f1_2))
        logging.info("K=" + str(args.pos_size[2]) + "Prec: " + str(max_p) + " Rec: " + str(max_r) + " F1: " + str(max_f1))
        accu, mle = evaluator.sparse_seq_predict(network, test_marker, test_time, test_mask, adj_list, 1)
        print("accu", accu)
        print("mle", mle)
        logging.info("accu: " + str(accu))
        logging.info("mle: " + str(mle))
        network.train()