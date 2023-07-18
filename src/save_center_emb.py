import torch
import json
import os


def save_tensor(t, dest):
    torch.save(t, dest)


if __name__ == '__main__':

    # Set the phase
    phase = 'test'
    train_test_path = os.path.join("../raw_data", "train_test.json")
    with open(train_test_path, 'r') as fp:
        train_test = json.load(fp)

    # All the embeddings
    all_emb_path = os.path.join("../processed_data", "mean_pooled_emb.pt")
    all_emb = torch.load(all_emb_path)  # [141245, 512]

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

    c_emb_list = []
    for a_id, a_graphs in all_part_graph.items():
        if a_id in train_test[phase]:
            for c_part, _ in a_graphs.items():
                c_idx = part_name_idx_map[c_part]
                c_emb_list.append(all_emb[c_idx])   # c_part is unique

    c_emb = torch.stack(c_emb_list)   # train: [42919, 512], test: [10377, 512]
    print(c_emb.shape)
    path = "../processed_data/center_emb_test.pt"
    save_tensor(c_emb, path)


