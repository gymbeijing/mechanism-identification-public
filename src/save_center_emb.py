import torch
import json
import os


def save_tensor(t, dest):
    torch.save(t, dest)


def save_json(data, dest):
    with open(dest, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


if __name__ == '__main__':

    # Set the phase
    phase = 'train'
    train_test_path = os.path.join("../raw_data", "train_test.json")
    with open(train_test_path, 'r') as fp:
        train_test = json.load(fp)

    # All the embeddings
    all_emb_path = os.path.join("../processed_data", "mean_pooled_emb.pt")
    all_emb = torch.load(all_emb_path)  # [141245, 512]
    pad = torch.zeros(1, 512)
    all_emb = torch.cat((all_emb, pad), 0)  # [141246, 512]

    # All the metadata
    all_md_path = os.path.join("../processed_data", "emb_idx_filtered.json")
    with open(all_md_path, 'r') as fp:
        all_md = json.load(fp)   # .shape[0]=141245
    all_md.append("pad")   # .shape[0]=141246

    # All the preprocessed part graphs
    all_part_graph_path = os.path.join("../processed_data", "part_graphs_dim=5.json")
    with open(all_part_graph_path, 'r') as fp:
        all_part_graph = json.load(fp)   # {a_id: {c1_md: [n_md]}}

    # Map part name to its idx in the metadata/embeddings
    part_name_idx_map = dict()
    for idx, p_name in enumerate(all_md):
        part_name_idx_map[p_name] = idx

    n_part_graph = 0
    for a_id, a_graphs in all_part_graph.items():
        if a_id in train_test[phase]:
            n_part_graph += len(a_graphs)

    c_emb_list = []
    aid_list = []
    cid_list = []
    for a_id, a_graphs in all_part_graph.items():
        if a_id in train_test[phase]:
            for c_part, _ in a_graphs.items():
                c_idx = part_name_idx_map[c_part]
                c_emb_list.append(all_emb[c_idx])   # c_part is unique
                aid_list.append(a_id)
                cid_list.append(c_idx)

    n_neighbour = 9
    all_data = torch.full((n_part_graph, n_neighbour + 1), fill_value=part_name_idx_map["pad"],
                               dtype=torch.long)  # initialize to be all pad's index

    # Fill in the matrix
    r = 0
    for a_id, a_graphs in all_part_graph.items():
        if a_id in train_test[phase]:
            for c_part, neigh_parts in a_graphs.items():
                all_data[r][0] = part_name_idx_map[c_part]
                c = 1
                for neigh_part in neigh_parts:
                    if c <= n_neighbour:  # only keep the first n_neighbour neighbouring parts
                        all_data[r][c] = part_name_idx_map[neigh_part]
                    else:
                        break
                    c += 1
                r += 1

    c_emb = torch.stack(c_emb_list)   # train: [42919, 512], test: [10377, 512]

    print(c_emb.shape)
    path = f"../processed_data/center_emb_{phase}.pt"
    save_tensor(c_emb, path)

    path = f"../processed_data/aid_{phase}.json"
    save_json(aid_list, path)

    path = f"../processed_data/cid_{phase}.json"
    save_json(cid_list, path)

    seq_emb = torch.zeros((all_data.shape[0], all_data.shape[1], 512))
    for i, row_indices in enumerate(all_data):
        seq_emb[i] = all_emb[row_indices]

    path = f"../processed_data/seq_emb_{phase}.pt"
    save_tensor(seq_emb, path)


