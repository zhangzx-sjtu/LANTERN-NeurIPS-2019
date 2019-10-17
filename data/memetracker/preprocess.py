import json
import time
import numpy as np
import torch


def check_cascades(time_list):
    # marker_list: a list with length len, reversed order
    if len(time_list) < 5:
        return 0
    length = len(time_list)
    success_or_not = 1
    init_time = np.inf
    for i in range(length):
        new_time = time_list[i]
        if new_time < init_time:
            init_time = new_time
        else:
            success_or_not = 0
            break
    return success_or_not


# Directories
data_dir = "quotes_2008-08.txt"
medium_dir_1 = "memetracker_1.txt"
medium_dir_2 = "memetracker_2.txt"
trace_file = "trace.json"
erased_trace_file = "erased_trace.json"

target_num = 50
total_target_cnt = 580


# Filtering out all "Q"s
with open(data_dir, "r", encoding="utf-8") as f:
    with open(medium_dir_1, "w", encoding="utf-8") as ff:
        print("Loading Raw Data ...")
        done = 0
        while not done:
        # for i in range(800000):
            line = f.readline()
            if line != '':
                if line != "\n":
                    line = line[0: -1]
                    if line[0] != 'Q':
                        ff.write(line)
                        ff.write('\n')
            else:
                done = 1
print("Done.")


# Get marker dictionary and continuous time stamps
marker_dict = {}
marker_time = {}
with open(medium_dir_1, "r", encoding="utf-8") as ff1:
    done = 0
    count = 0
    print("Building Dictionary ...")
    while not done:
        line = ff1.readline()[0: -1]
        if line != '':
            if line[0] == 'P':
                split_line = line.split("\t")
                marker_dict.update({split_line[-1]: count})

                time_line = ff1.readline()[0: -1]
                time_date = time_line.split("\t")[-1]
                time_array = time.strptime(time_date, "%Y-%m-%d %H:%M:%S")
                real_time = time.mktime(time_array)

                marker_time.update({str(count): real_time})
                count += 1
            else:
                pass
        else:
            done = 1
print("Done.")

print("Building JSON Datasets ...")
# Get the trace of memetracker
with open(medium_dir_1, "r", encoding="utf-8") as fff1:
    with open(trace_file, "w", encoding="utf-8") as fff2:
        done = 0
        already_read = 0
        while not done:
            if not already_read:
                line = fff1.readline()[0: -1]
            # print(line)
            if line != '':
                if line[0] == 'P':
                    data_dict = {}
                    # fill in the marker type
                    split_line = line.split("\t")
                    # print("split_line", split_line)
                    data_dict.update({"marker": marker_dict[split_line[-1]]})
                    # fill in the real-time
                    time_line = fff1.readline()[0: -1]
                    split_time = time_line.split("\t")[-1]
                    time_array = time.strptime(split_time, "%Y-%m-%d %H:%M:%S")
                    real_time = time.mktime(time_array)
                    data_dict.update({"time": real_time})
                    # fill in the ref
                    ref_list = []
                    while True:
                        possible_l = fff1.readline()
                        if possible_l == "":
                            done = 1
                            break
                        else:
                            possible_l = possible_l[0: -1]
                            if possible_l[0] == "P":
                                line = possible_l
                                already_read = 1
                                break
                            else:
                                split_l = possible_l.split("\t")[-1]
                                if not (split_l in marker_dict):
                                    continue
                                else:
                                    ref_list.append(marker_dict[split_l])
                    data_dict.update({"ref": ref_list})

                    if data_dict["ref"] != []:
                        fff2.write(json.dumps(data_dict))
                        fff2.write("\n")
            else:
                done = 1


# From the trace file get the cascades
load_markers = []
load_refs_length = []
with open(trace_file, "r", encoding="utf-8") as ddt:
    with open(erased_trace_file, "w", encoding="utf-8") as eddt:
        done = 0
        while not done:
            line = ddt.readline()
            if line != "":
                data_dict = json.loads(line)
                new_ref_list = []
                for i in range(len(data_dict["ref"])):
                    if data_dict["ref"][i] < data_dict["marker"]:
                        new_ref_list.append(data_dict["ref"][i])
                data_dict.update({"ref": new_ref_list})
                if new_ref_list != []:
                    load_markers.append(data_dict["marker"])
                    load_refs_length.append(len(data_dict["ref"]))
                    eddt.write(json.dumps(data_dict))
                    eddt.write("\n")
            else:
                done = 1


np_load_markers = np.array(load_markers)
np_load_refs_length = np.array(load_refs_length)

index = np.argsort(np_load_refs_length)
# print(index)
sorted_markers = np.zeros(len(load_markers), dtype=np.int32)

for i in range(len(load_markers)):
    sorted_markers[i] = np_load_markers[index[i]]

# print(sorted_markers)
sorted_markers = sorted_markers.tolist()
sorted_markers.reverse()

# Build Dictionary
searching_dict = {}
with open(erased_trace_file, "r", encoding="utf-8") as file:
    done = 0
    while not done:
        line = file.readline()
        if line != '':
            data_dict = json.loads(line)
            searching_dict.update({str(data_dict["marker"]): data_dict["ref"]})
        else:
            done = 1
print("Done.")

# Tracing and get data
print("Sampling Data ...")
cascade_num = 0
node_num = 0

data_markers = torch.zeros(0, 6, dtype=torch.long)
data_times = torch.zeros(0, 6, dtype=torch.float)
data_masks = torch.zeros(0, 6, dtype=torch.float)

sel_marker = torch.LongTensor([[0, 0, 0, 0, 0, 0]])
sel_time = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])
sel_mask = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

sel_marker_list = []
sel_time_list = []
sel_mask_list = []

selected_marker_dict = {}
marker_count = 0
time_to_break = 0

# print(selected_marker_dict)
for i in range(len(sorted_markers)):
    print(i)
    current_num = 0
    while current_num < target_num:
        marker_root = sorted_markers[i]
        sel_marker_list.append(marker_root)
        sel_time_list.append(marker_time[str(marker_root)])
        sel_mask_list.append(1.0)

        while True:
            if (not(str(marker_root) in searching_dict)) or (len(sel_marker_list) == 6):
                break
            else:
                next_marker_list = searching_dict[str(marker_root)]
                length = len(next_marker_list)
                prob_sample = torch.ones(1, length)
                chosen_index = torch.multinomial(prob_sample, 1).long().item()
                next_marker = next_marker_list[chosen_index]

                sel_marker_list.append(next_marker)
                sel_time_list.append(marker_time[str(next_marker)])
                sel_mask_list.append(1.0)

                marker_root = next_marker

        for i in range(len(sel_time_list)):
            sel_time_list[i] = sel_time_list[i] - 1217520000
        succ = check_cascades(sel_time_list)

        if succ:
            for marker in sel_marker_list:
                if not (str(marker) in selected_marker_dict):
                    selected_marker_dict.update({str(marker): str(marker_count)})
                    marker_count += 1

            curr_length = len(sel_time_list)
            sel_marker[0, 0: curr_length] = torch.LongTensor(sel_marker_list[::-1])
            sel_time[0, 0: curr_length] = torch.FloatTensor(sel_time_list[::-1])
            sel_mask[0, 0: curr_length] = torch.FloatTensor(sel_mask_list[::-1])
            data_markers = torch.cat((data_markers, sel_marker.clone()), 0)
            data_times = torch.cat((data_times, sel_time.clone()), 0)
            data_masks = torch.cat((data_masks, sel_mask.clone()), 0)

            sel_marker = torch.LongTensor([[0, 0, 0, 0, 0, 0]])
            sel_time = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])
            sel_mask = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        current_num += 1

        sel_marker_list = []
        sel_time_list = []
        sel_mask_list = []
        print("m", marker_count)
        if marker_count > total_target_cnt:
            time_to_break = 1
            break
    if time_to_break:
        print("Break!")
        break

reversed_selected_marker_dict = dict(zip(selected_marker_dict.values(), selected_marker_dict.keys()))

length_list = torch.sum(data_masks, 1).long().tolist()
data_size = data_markers.size()

for i in range(data_size[0]):
    for j in range(length_list[i]):
        marker = str(data_markers[i][j].item())
        data_markers[i][j] = int(selected_marker_dict[marker])

total_marker_num = len(selected_marker_dict)
network_adjacent = torch.zeros(total_marker_num, total_marker_num)

print("Building Networks ...")
for i in range(total_marker_num):
    real_marker = reversed_selected_marker_dict[str(i)]
    if not (real_marker in searching_dict):
        continue
    else:
        fathers_list = searching_dict[real_marker]
    for item in fathers_list:
        if not (str(item) in selected_marker_dict):
            continue
        else:
            real_index = int(selected_marker_dict[str(item)])
            network_adjacent[real_index][i] = 1.0

adj_list = []
for i in range(total_marker_num):
    adj_i = []
    for j in range(total_marker_num):
        if network_adjacent[i][j] == 1:
            adj_i.append(j)
    adj_list.append(adj_i)

data_times = data_masks * (data_times - data_times[:, 0:1])
max_time = torch.max(data_times)
data_times = (data_times / max_time) * 5

print("Done.")
print("Saving pkl files ...")
torch.save(data_markers, "marker.pkl")
torch.save(data_times, "time.pkl")
torch.save(data_masks, "mask.pkl")
torch.save(network_adjacent, "adj_mat.pkl")
torch.save(adj_list, "adj_list.pkl")
print("Done.")