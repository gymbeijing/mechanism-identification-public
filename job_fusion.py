import torch
import json
import os

from data.ae_dataset import AEDataset
from torch import linalg as LA
import numpy as np
from matplotlib.pyplot import plot
from tqdm import tqdm
import random


def save_tensor(t, filepath):
    torch.save(t, filepath)
    return


def load_saved_tensor(filepath):
    t = torch.load(filepath)

    return t


def load_json(filepath):
    with open(filepath, 'r') as fp:
        j = json.load(fp)  # .shape[0]=141245

    return j


def compute_aid_to_part_indice_map():
    the_map = dict()
    assert len(all_md) == all_emb.shape[0]
    for idx, md in enumerate(all_md):
        aid = '_'.join(md.split('_')[1:3])
        if aid in the_map:
            the_map[aid].append(idx)
        else:
            the_map[aid] = []
            the_map[aid].append(idx)

    return the_map


def compute_md_to_idx_map():
    the_map = dict()
    for idx, md in enumerate(all_md):
        the_map[md] = idx

    return the_map


def compute_norm(seq):
    all_norm_list = []
    for s in seq:
        # seq: [10, 512]
        norm_list = []
        for emb in s:
            # emb: [512]
            norm_list.append(LA.norm(emb).item())
        all_norm_list.append(norm_list)

    return all_norm_list


def compute_norm_clip_emb(seq):
    all_norm_list = []
    for emb in seq:
        all_norm_list.append(LA.norm(emb).item())

    return all_norm_list


def decode_seq_3_fast(in_seq, head_number, all_norm_list):
    all_decoded_seq_list = []
    for i, seq in tqdm(enumerate(in_seq[:head_number, :, :])):
        seq_dist = []
        for j, rec_emb in enumerate(seq):
            delta = all_emb - rec_emb  # [141245, 512]
            emb_dist = LA.norm(delta, dim=1)
            seq_dist.append(emb_dist)

        seq_dist = torch.stack(seq_dist, dim=1)  # [141245, 10]
        all_decoded_seq_dict = dict()
        len_indices_equals_zero_flag = False
        for aid in valid_aid_list:
            indices = aid_to_part_indice_map[aid]
            decoded_seq = []
            indices = indices.copy()
            total_dist = 0.0
            total = 0
            # sequential search
            for j, rec_emb in enumerate(seq):
                if all_norm_list[i][j] < eps:  # if norm < eps, stop decoding, should check on the first(central) part?
                    break
                if len(indices) == 0:
                    len_indices_equals_zero_flag = True
                    break
                min_val = torch.min(torch.index_select(seq_dist[:, j], 0, torch.LongTensor(indices))).item()
                idx_tuple = (seq_dist == min_val).nonzero(as_tuple=True)[
                    0]  # first [0]: (tensor([91286]),) second[0]: remove multiple values
                for idx in idx_tuple:  # will find one for sure
                    if idx in indices:
                        decoded_seq.append(idx.item())
                        total_dist += seq_dist[idx, j].item()
                        break
                indices.remove(idx)
                total += 1
            if not len_indices_equals_zero_flag:
                all_decoded_seq_dict[aid] = (decoded_seq, total_dist / total if total != 0 else 100000)
        all_decoded_seq_list.append(all_decoded_seq_dict)

    return all_decoded_seq_list


def decode_seq_3_random(in_seq, head_number, all_norm_list):
    all_decoded_seq_list = []
    for i, seq in tqdm(enumerate(in_seq[:head_number, :, :])):
        seq_dist = []
        for j, rec_emb in enumerate(seq):
            delta = all_emb - rec_emb  # [141245, 512]
            emb_dist = LA.norm(delta, dim=1)
            seq_dist.append(emb_dist)

        seq_dist = torch.stack(seq_dist, dim=1)  # [141245, 10]
        all_decoded_seq_dict = dict()
        len_indices_equals_zero_flag = False
        for aid in valid_aid_list:
            indices = aid_to_part_indice_map[aid]
            decoded_seq = []
            indices = indices.copy()
            total_dist = 0.0
            total = 0
            # sequential search
            for j, rec_emb in enumerate(seq):
                if all_norm_list[i][j] < eps:  # if norm < eps, stop decoding, should check on the first(central) part?
                    break
                if len(indices) == 0:
                    len_indices_equals_zero_flag = True
                    break
                idx = random.choice(indices)
                decoded_seq.append(idx)
                total_dist += seq_dist[idx, j].item()
                indices.remove(idx)
                total += 1
            if not len_indices_equals_zero_flag:
                all_decoded_seq_dict[aid] = (decoded_seq, total_dist / total if total != 0 else 100000)
        all_decoded_seq_list.append(all_decoded_seq_dict)

    return all_decoded_seq_list


if __name__ == '__main__':
    rec_seq_path = "model_outputs/rec_seq_0801_161839_last.pt"

    rec_seq = load_saved_tensor(rec_seq_path)  # [10377, 10, 512]

    all_emb_path = os.path.join("./processed_data", "mean_pooled_emb.pt")
    all_emb = load_saved_tensor(all_emb_path)  # [141245, 512]

    all_md_path = os.path.join("./processed_data", "emb_idx_filtered.json")
    all_md = load_json(all_md_path)  # .shape[0]=141245

    aid_to_part_indice_map = compute_aid_to_part_indice_map()  # 8146: # assemblies got successfully processed/rendered

    print(f'Total number of assemblies: {len(aid_to_part_indice_map)}')

    md_to_idx_map = compute_md_to_idx_map()
    print(f'Number of parts: {len(md_to_idx_map)}')

    all_part_graph_path = './processed_data/part_graphs_dim=5.json'
    all_part_graph = load_json(all_part_graph_path)
    valid_aid_list = list(
        all_part_graph.keys())  # 5462: # assemblies having contact info, also # assemblies in the part graph
    print(f'Number of assemblies having contact information / included in the part graph: {len(valid_aid_list)}')

    perm = torch.randperm(rec_seq.size(0))
    indices = perm[:1000]
    torch.save(indices, './model_outputs/rand_indices_1000_for_fusion_disposed.pt')
    rec_seq_sample = rec_seq[indices]

    all_rec_norm_list = compute_norm(rec_seq_sample)

    eps = 0.5
    threshold = 0.5

    print(f'Number of data items: {rec_seq_sample.shape[0]}')
    all_decoded_seq_list = decode_seq_3_random(rec_seq, rec_seq_sample.shape[0], all_rec_norm_list)
    with open(f'./model_outputs/rec_decoded_seq_list_3_random_1000_disposed.json', 'w', encoding='utf8') as fp:  # maybe filtered_assembly_ids.json is a better name
        json.dump(all_decoded_seq_list, fp, indent=4, ensure_ascii=False, sort_keys=False)

