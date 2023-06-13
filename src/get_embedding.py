import torch
from torch.utils.data import Dataset
from lavis.models import load_model_and_preprocess
import argparse
import glob
from PIL import Image
import torch.utils.data as data
from tqdm import tqdm
import json


class PartImageDataset(Dataset):
    def __init__(self, image_folder, model, vis_processors, txt_processors, device):
        self.image_folder = image_folder
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.image_list = glob.glob(self.image_folder + '/*.png')[:1000]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        raw_image = Image.open(img_name)
        caption = ""
        image = vis_processors["eval"](raw_image).to(device)   # .unsqueeze(0): [1, 3, 224, 224], otherwise: [3, 224, 224]
        text_input = txt_processors["eval"](caption)   # strings

        img_id = img_name.split('/')[-1].split('.')[0]  # remove the .png from the image name

        return img_id, image, text_input


def get_emb(dataloader):
    total_img_id_list = []
    batch_emb_list = []
    for i, (batch_img_id, batch_image, batch_text_input) in tqdm(enumerate(imageloader, 0)):
        samples = {"image": batch_image, "text_input": list(batch_text_input)}
        features = model.extract_features(samples)
        features_image = features.image_embeds   # [bs, 512]

        total_img_id_list += list(batch_img_id)
        batch_emb_list.append(features_image)

    total_emb = torch.cat(batch_emb_list, dim=0)

    return total_img_id_list, total_emb


def save_emb_to_file(ids, embs, dirname, vid):
    print(embs.shape)
    print(f'Saving embeddings to {dirname}/clip_emb_{vid}.pt')
    torch.save(embs, dirname+'/'+f'clip_emb_{vid}.pt')
    print(len(ids))
    print(f'Saving embeddings to {dirname}/emb_idx_{vid}.json')
    # with open(dirname+'/'+f'emb_idx_{vid}.txt', 'w') as f:
    #     for index in ids:
    #         f.write(f"{index}\n")
    with open(dirname+'/'+f'emb_idx_{vid}.json', 'w', encoding='utf8') as fp:
        json.dump(ids, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, help='Path to the directory that saves part images')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model, vis_processors, txt_processors = load_model_and_preprocess(name='clip_feature_extractor', model_type="base",
                                                                      is_eval=True, device=device)
    # print(vis_processors)

    # Instantiate the dataset and the dataloader
    images = PartImageDataset(args.image_folder, model, vis_processors, txt_processors, device)
    imageloader = data.DataLoader(images, shuffle=False, batch_size=32)

    image_ids, image_embs = get_emb(imageloader)

    vid = args.image_folder.split('/')[-1]
    save_emb_to_file(image_ids, image_embs, '../emb', vid)
