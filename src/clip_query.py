import json
from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset
import torch.utils.data as data


def load_json(path):
    with open(path) as fp:
        data = json.load(fp)

    return data


class PartQueryDataset(Dataset):
    def __init__(self, building_blocks, vis_processors, txt_processors):
        self.building_blocks_list = []
        for mech, info in building_blocks.item():
            self.building_blocks_list.append((mech, info["must_include"]))

        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.building_blocks_list)

    def __getitem__(self, idx):
        return self.building_blocks_list[idx]


if __name__ == '__main__':
    building_blocks = load_json('../raw_data/building_blocks_list.json')   # 26
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name='clip_feature_extractor', model_type="base",
                                                                      is_eval=True, device=device)

    # Instantiate the dataset and the dataloader
    queries = PartQueryDataset(building_blocks, vis_processors, txt_processors)
    queryloader = data.DataLoader(queries, shuffle=False, batch_size=64)
