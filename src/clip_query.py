import json
from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from tqdm import tqdm
import torch.nn.functional as F


def load_json(path):
    with open(path) as fp:
        data = json.load(fp)

    return data


def get_emb(dataloader, model):
    """
    Get the ids and the embeddings from the dataloader
    :param dataloader: batch load the images
    :param model: the encoder, e.g. CLIP
    :return: the list of the image embedding ids and the concatenated image embeddings
    """
    all_mech_name_list = []
    all_part_name_list = []
    batch_emb_list = []
    for i, (batch_mech_name, batch_part_name, batch_text_input) in tqdm(enumerate(dataloader, 0)):
        samples = {"text_input": list(batch_text_input)}
        text_embeds = model.extract_features(samples)
        text_features = F.normalize(text_embeds, dim=-1)  # [bs, 512], image_embeds_proj is normalized image_embeds

        all_mech_name_list += list(batch_mech_name)
        all_part_name_list += list(batch_part_name)
        batch_emb_list.append(text_features.detach().cpu())   # .detach().cpu() is to free up gpu memory, otherwise will run into OOM issue

    all_emb = torch.cat(batch_emb_list, dim=0)

    return all_mech_name_list, all_part_name_list, all_emb


def save_tensor(tensor, path):
    torch.save(tensor, path)

    return


def save_json(dict, path):
    with open(path, 'w', encoding='utf8') as fp:
        json.dump(dict, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return

class PartQueryDataset(Dataset):
    def __init__(self, building_blocks, txt_processors):
        self.building_blocks_list = []
        for mech_name, info in building_blocks.items():
            must_include = info.get("must_include")
            if must_include is not None:
                for part_name in must_include:
                    self.building_blocks_list.append((mech_name, part_name))

        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.building_blocks_list)

    def __getitem__(self, idx):
        mech_name = self.building_blocks_list[idx][0]
        part_name = self.building_blocks_list[idx][1]
        text_input = txt_processors["eval"](part_name)

        return mech_name, part_name, text_input


if __name__ == '__main__':
    building_blocks = load_json('../raw_data/building_blocks_list.json')   # 26
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name='clip_feature_extractor', model_type="base",
                                                                      is_eval=True, device=device)

    # Instantiate the dataset and the dataloader
    queries = PartQueryDataset(building_blocks, txt_processors)   # 30
    queryloader = data.DataLoader(queries, shuffle=False, batch_size=64)

    mech_names, part_names, embs = get_emb(queryloader, model)

    mech_part_dict = dict()
    mech_part_dict["mech_name"] = mech_names
    mech_part_dict["part_name"] = part_names

    save_tensor(embs, '../processed_data/clip_query_embs.pt')
    save_json(mech_part_dict, '../processed_data/clip_query_metadata.json')
    print(mech_names)
    print(part_names)
    print(embs.shape)
