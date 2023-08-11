import torch
import json
import os

from data.ae_dataset import AEDataset
from torch import linalg as LA
import numpy as np
from matplotlib.pyplot import plot
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Markdown, display_markdown # to display images

import numpy as np
import random
from matplotlib import pyplot as plt
import random


def compute_md_to_idx_map():
    the_map = dict()
    for idx, md in enumerate(all_md):
        the_map[md] = idx

    return the_map


def load_saved_tensor(filepath):
    t = torch.load(filepath)

    return t


def save_json(data, path):
    with open(path, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


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

def transform(part_graph_dict):
    returned_dict = dict()
    for c_md, n_md_list in part_graph_dict.items():
        idx_list = []
        for n_md in n_md_list:
            idx_list.append(md_to_idx_map[n_md])

        returned_dict[md_to_idx_map[c_md]] = idx_list

    return returned_dict


def compute_iou(y_pred, y_true, dup_map, epsilon=1e-6):
    y_pred = set(y_pred)
    y_true = set(y_true)
    len_pred = len(y_pred)
    len_true = len(y_true)

    assert len_pred > 0
    #     print(dup_map)

    #     intersection = y_pred.intersection(y_true)

    intersection = []
    #     print(f"y_true: {y_true}")
    #     print(f"y_pred: {y_pred}")
    #     print(f"dup_map: {dup_map}")
    for part_idx_true in y_true:
        if len(intersection) == min(len_pred, len_true):
            break
        for part_idx_pred in y_pred:
            if part_idx_true in dup_map[part_idx_pred]:
                intersection.append(part_idx_true)
                break  # otherwise will append multiple times, which is wrong

    union = y_pred.union(y_true)
    #     print(f"intersection: {intersection}")

    iou_rec = len(intersection) / len_pred
    iou_prec = len(intersection) / len_true
    iou_score = len(intersection) / len(union)
    iou_avg = 2 * iou_prec * iou_rec / (iou_prec + iou_rec + epsilon)

    assert iou_rec <= 1, f"IOU recall is greater than 1: {iou_rec}, intersection: {len(intersection)}, len_pred: {len_pred}"
    assert iou_prec <= 1, f"IOU precision is greater than 1: {iou_prec}"
    assert iou_score <= 1, f"IOU score is greater than 1: {iou_score}"
    assert iou_avg <= 1, f"IOU average is greater than 1: {iou_avg}"

    return iou_avg, iou_score  # , len(union)


def search_duplicate(c_part_idx, aid, eps=1e-3):
    indices = aid_to_part_indice_map[aid]  # all the indices belong to aid
    start_idx = indices[0]  # the start index of aid in all_emb
    embs = all_emb[indices]  # embeddings of aid
    cdist = torch.cdist(embs, embs, p=2,
                        compute_mode="donot_use_mm_for_euclid_dist")  # distance matrix of all the embeddings
    indices = torch.LongTensor(indices)  # make indices ready for indexing
    # get the row index corresponding to the c_part_idx
    c_part_dup_list = indices[
        torch.LongTensor((cdist[c_part_idx - start_idx] <= eps).nonzero(as_tuple=True)[0])].tolist()

    return c_part_dup_list


def init_map():
    the_map = dict()
    for valid_aid in valid_aid_list:
        the_map[valid_aid] = []

    return the_map


def eval_iou(decoded_seq_list, k=10):
    max_max_iou_list = []
    max_max_iou_score_list = []
    len_topk_a = []
    for i, decoded_seq_dict in enumerate(tqdm(decoded_seq_list)):
        ordered_decoded_seq = sorted(decoded_seq_dict.items(), key=lambda x: x[1][1])  # [(aid, (seq_list, avg_dist))]
        ordered_topk_decoded_seq = ordered_decoded_seq[:k]  # top k avg_dist [(aid, (seq_list, avg_dist))]
        max_max_iou = 0.0
        max_max_iou_score = 0.0
        kept_a = None
        for aid, (seq, avg_dist) in ordered_topk_decoded_seq:

            aid_part_graph = all_part_graph[aid]  # obtain all the part graphs associated with aid
            aid_part_graph = transform(aid_part_graph)  # map list of md --> list of idx
            c_part = seq[0]  # central part idx
            n_parts_pred = seq[1:]  # a list of neighbouring part indices
            max_iou = 0  # we only record the max_iou of all the sequence considering duplicates
            max_iou_score = 0

            # Finding duplicates for each part in the decoded seq
            dup_map = dict()
            for part_idx in seq:
                dup_map[part_idx] = search_duplicate(part_idx, aid)

            for c_part in dup_map[c_part]:
                if c_part in aid_part_graph:  # if aid_part_graph contains c_part in the decoded seq, otherwise max_iou stays 0
                    n_parts_true = aid_part_graph[c_part][:9]

                    iou, iou_score = compute_iou(n_parts_pred, n_parts_true, dup_map)
                    if iou > max_iou:
                        max_iou = iou
                    if iou_score > max_iou_score:
                        max_iou_score = iou_score

            if max_iou > max_max_iou:
                max_max_iou = max_iou  # compute 10 times
            if max_iou_score > max_max_iou_score:
                max_max_iou_score = max_iou_score
                kept_a = aid

            # result["num_nodes"] = len(aid_to_part_indice_map[aid])
            # result["seq_length"] = 1 + len(n_parts_pred)
            # result["k"] = k
            # result["iou_score"] = max_iou_score
            key = (aid, len(aid_to_part_indice_map[aid]), 1 + len(n_parts_pred), k)
            if key in result:
                result[key].append(max_iou_score)
            else:
                result[key] = []
                result[key].append(max_iou_score)

        if kept_a is None:
            len_topk_a.append(0)
        else:
            len_topk_a.append(len(aid_to_part_indice_map[kept_a]))

        max_max_iou_list.append(max_max_iou)
        max_max_iou_score_list.append(max_max_iou_score)

    return max_max_iou_list, max_max_iou_score_list, len_topk_a


def reformat_result():
    for (aid, num_nodes, seq_length, k), iou_list in result.items():
        temp = dict()
        temp["num_nodes"] = num_nodes
        temp["seq_length"] = seq_length
        temp["k"] = k
        # temp["iou_score"] = np.array(iou_list).mean()
        temp["iou_score"] = iou_list
        aid_result_map[aid].append(temp)

    return


if __name__ == '__main__':

    all_emb_path = os.path.join("./processed_data", "mean_pooled_emb.pt")
    all_emb = load_saved_tensor(all_emb_path)  # [141245, 512]

    all_md_path = os.path.join("./processed_data", "emb_idx_filtered.json")
    all_md = load_json(all_md_path)  # .shape[0]=141245

    md_to_idx_map = compute_md_to_idx_map()
    print(f'Number of parts: {len(md_to_idx_map)}')

    aid_to_part_indice_map = compute_aid_to_part_indice_map()  # 8146: # assemblies got successfully processed/rendered

    all_part_graph_path = './processed_data/part_graphs_dim=5.json'
    all_part_graph = load_json(all_part_graph_path)
    valid_aid_list = list(
        all_part_graph.keys())  # 5462: # assemblies having contact info, also # assemblies in the part graph
    print(f'Number of assemblies having contact information / included in the part graph: {len(valid_aid_list)}')

    # Sampled Fusion360 test set
    all_decoded_seq_list = load_json('./model_outputs/rec_decoded_seq_list_3_1000_disposed.json')

    aid_result_map = init_map()
    result = dict()

    res = []
    klist = [1, 5, 10]
    for k in klist:
        res.append(eval_iou(all_decoded_seq_list, k))

    reformat_result()

    save_json(aid_result_map, './model_outputs/iou_list_by_assembly.json')

    # res = []
    # k = 1
    # print(f'k={k}')
    # res.append(eval_iou(all_decoded_seq_list, k))
    #
    # length_iou_map = dict()
    # for i, length in enumerate(res[0][2]):
    #     if length not in length_iou_map:
    #         length_iou_map[length] = []
    #         length_iou_map[length].append(res[0][1][i])
    #     else:
    #         length_iou_map[length].append(res[0][1][i])
    #
    # length_mean_iou_map = dict()
    # for length, iou_list in length_iou_map.items():
    #     length_mean_iou_map[length] = sum(iou_list) / len(iou_list)
    #
    # print(length_mean_iou_map)
    print(f'k = 1: avg_iou: {np.array(res[0][0]).mean()}, avg_iou_score: {np.array(res[0][1]).mean()}')
    print(f'k = 5: avg_iou: {sum(res[1][0]) / len(res[1][0])}, avg_iou_score: {sum(res[1][1]) / len(res[1][1])}')
    print(f'k = 10: avg_iou: {sum(res[2][0]) / len(res[2][0])}, avg_iou_score: {sum(res[2][1]) / len(res[2][1])}')

