import torch
import json
import os
from torch import linalg as LA
from tqdm import tqdm
import random
import argparse
from pathlib import Path


def save_tensor(t, dest):
    torch.save(t, dest)


def load_tensor(src):
    tensor = torch.load(src)
    return tensor


def load_json(src):
    with open(src, 'r') as fp:
        json_data = json.load(fp)
    return json_data


def save_json(data, dest):
    with open(dest, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


def retrieve_sequence_indices(queries, keys, eps, k):
    k_indices_list = []
    for query in tqdm(queries):
        delta = keys - query
        dist = LA.norm(delta, dim=1)
        values, indices = torch.sort(dist, descending=False)   # in ascending order
        indices = indices[values<=eps]
        if indices.size(0) >= k:
            k_indices_list.append(torch.LongTensor(random.sample(indices.tolist(), k)))
        else:
            k_indices_list.append(None)
    return k_indices_list


def get_c_emb(seq):
    return seq[:, 0, :]   # first vector is the central emb


def get_decoded_sequences(seq, k_indices_list, k):
    nothing_decoded = torch.zeros(10, 512)
    k_decoded_sequence_list = []
    for i in range(k):
        k_decoded_sequence_list.append([])

    for k_indices in k_indices_list:   # k_indices: a tensor
        if k_indices is None:
            for i in range(k):
                k_decoded_sequence_list[i].append(nothing_decoded)
        else:
            for i, index in enumerate(k_indices):
                k_decoded_sequence_list[i].append(seq[index])

    concat_k_decoded_sequence_list = []
    for i in range(k):
        concat_k_decoded_sequence_list.append(torch.stack(k_decoded_sequence_list[i]))
    return concat_k_decoded_sequence_list


def get_none_indices(in_list):
    out_list = []
    for idx, item in enumerate(in_list):
        if item is None:
            out_list.append(idx)

    return out_list


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_test", type=str, default="../raw_data/train_test.json", required=True,
                   help="Json file that contains the train test split")
    p.add_argument("--seq_emb_train", type=str, default="../model_outputs/ori_seq_0824_135617_last_train.pt",
                   required=True, help="Part sequence embeddings of the training set")
    p.add_argument("--c_emb_test", type=str, default="../processed_data/center_emb_test.pt", required=True,
                   help="Central embeddings of the test set")
    p.add_argument("--k", type=int, default=1, required=True,
                   help="To retrieve k closest part sequences from the training set")
    p.add_argument("--eps", type=float, default=1e-3, required=True,
                   help="Radius of the epsilon ball")
    p.add_argument("--output_folder", type=str, default="../model_outputs/1_retrieve_seq/", required=True,
                   help="Folder to the output .pt files")

    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    train_test_path = Path(args.train_test)
    train_test = load_json(train_test_path)

    seq_emb_train_path = Path(args.seq_emb_train)
    seq_emb_train = load_tensor(seq_emb_train_path)   # [42919, 10, 512] shuffled, shuffled or not doesn't matter

    # Get all c in train
    c_emb_train = get_c_emb(seq_emb_train)   # [42919, 512]

    # Get all c in test
    c_emb_test_path = Path(args.c_emb_test)   # unshuffled
    c_emb_test = load_tensor(c_emb_test_path)   # [10377, 512]

    eps = args.eps
    k = args.k
    retrieved_indices_list = retrieve_sequence_indices(c_emb_test, c_emb_train, eps, k)   # 10377

    none_indices_list = get_none_indices(retrieved_indices_list)
    print(len(none_indices_list))

    save_json({"None": none_indices_list}, "../model_outputs/none_indices_in_test.json")

    # decoded_sequences = get_decoded_sequences(seq_emb_train, retrieved_indices_list, k)
    #
    # for i in range(k):
    #     dest = os.path.join(args.output_folder, f"retrieve_seq_{i}.pt")
    #     assert decoded_sequences[i].shape == torch.Size([10377, 10, 512])
    #     save_tensor(decoded_sequences[i], dest)



