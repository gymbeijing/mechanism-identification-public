import torch
import json
import os


if __name__ == '__main__':
    test_aid_c_map_list = []
    train_test_path = os.path.join("./raw_data", "train_test.json")
    with open(train_test_path, 'r') as fp:
        train_test = json.load(fp)

    all_part_graph_path = os.path.join("./processed_data", "part_graphs_dim=5.json")
    with open(all_part_graph_path, 'r') as fp:
        all_part_graph = json.load(fp)

    for a_id, a_graphs in all_part_graph.items():
        if a_id in train_test["test"]:
            for c_part, neigh_parts in a_graphs.items():
                test_aid_c_map_list.append({"aid": a_id, "c_part_md": c_part})

    test_aid_c_map_list_path = os.path.join("./processed_data", "test_aid_c_map_list.json")
    with open(test_aid_c_map_list_path, 'w', encoding='utf8') as fp:
        json.dump(test_aid_c_map_list, fp, indent=4, ensure_ascii=False, sort_keys=False)
