import torch
import pickle

with open("repost_data.txt", "r", encoding="utf-8") as f:
    marker_data = torch.zeros(100000, 10, dtype=torch.long)
    time_data = torch.zeros(100000, 10, dtype=torch.float)
    mask_data = torch.zeros(100000, 10, dtype=torch.float)

    cascade_num = 0

    while cascade_num < 100000:
        command_line = f.readline().split("\t")

        id = int(command_line[0])
        total_num = int(command_line[1].strip())
        start_time = 0
        exist_num = 0
        flag = 0

        for i in range(total_num):
            line = f.readline().split("\t")
            time_stamp = float(line[0])
            marker = int(line[1].strip())

            if i == 0:
                start_time = time_stamp

            if marker >= 0 and marker < 100000 and exist_num < 10:
                marker_data[cascade_num][exist_num] = marker
                time_data[cascade_num][exist_num] = time_stamp - start_time
                mask_data[cascade_num][exist_num] = 1
                exist_num += 1

        if exist_num != 0:
            print(exist_num)

        if exist_num > 1:
            cascade_num += 1
        else:
            marker_data[cascade_num] = torch.zeros(1, 10).squeeze(0)
            time_data[cascade_num] = torch.zeros(1, 10).squeeze(0)
            mask_data[cascade_num] = torch.zeros(1, 10).squeeze(0)

    max_time = torch.max(time_data)
    time_data = time_data / max_time * 20
    torch.save(marker_data, "marker.pkl")
    torch.save(time_data, "time.pkl")
    torch.save(mask_data, "mask.pkl")

with open("weibo_network.txt", "r", encoding="utf-8") as f:

    adjacent = [[] for _ in range(100000)]
    f.readline()
    for i in range(100000):
        adjacent[i].append(i)
        line = f.readline().split("\t")
        line[-1] = line[-1].strip()
        length = len(line)
        j = 2

        while j < length:
            index = int(line[j])
            if index < 100000:
                adjacent[i].append(index)
            j += 2

    f = open("adjacent.pkl", "wb")
    pickle.dump(adjacent, f)


