import torch
from torch.utils.data import Dataset
from lavis.models import load_model_and_preprocess
import argparse
import glob
from PIL import Image
import torch.utils.data as data
from tqdm import tqdm
import json


class ABCPartImageDataset(Dataset):
    def __init__(self, body_id_list, vid, vis_processors, txt_processors, device):
        self.body_id_list = body_id_list
        self.vid = vid
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.device = device

    def __len__(self):
        return len(self.body_id_list)

    def __getitem__(self, idx):
        body_id = self.body_id_list[idx]
        img_id = body_id + '_00' + self.vid
        img_name = body_id + '_00' + self.vid + '.png'
        raw_image = Image.open(img_name)
        caption = ""
        image = vis_processors["eval"](raw_image).to(
            self.device)  # .unsqueeze(0): [1, 3, 224, 224], otherwise: [3, 224, 224]
        text_input = txt_processors["eval"](caption)  # strings

        return img_id, image, text_input


def get_body_id(in_list):
    out = []
    for img_name in in_list:
        body_id = '_'.join('/'.join(img_name.split('/')[-3:]).split('_')[:-1])  # keep the chunk name
        if body_id not in out:
            out.append(body_id)

    return out


def get_emb(dataloader, model):
    """
    Get the ids and the embeddings from the dataloader
    :param dataloader: batch load the images
    :param model: the encoder, e.g. CLIP
    :return: the list of the image embedding ids and the concatenated image embeddings
    """
    all_img_id_list = []
    batch_emb_list = []
    for i, (batch_img_id, batch_image, batch_text_input) in tqdm(enumerate(dataloader, 0)):
        samples = {"image": batch_image, "text_input": list(batch_text_input)}
        features = model.extract_features(samples)
        features_image = features.image_embeds_proj  # [bs, 512], image_embeds_proj is normalized image_embeds

        all_img_id_list += list(batch_img_id)
        batch_emb_list.append(
            features_image.detach().cpu())  # .detach().cpu() is to free up gpu memory, otherwise will run into OOM issue

    all_emb = torch.cat(batch_emb_list, dim=0)

    return all_img_id_list, all_emb


def save_emb(ids, embs, dirname, vid):
    """
    Save embeddings to file
    :param ids: the list of the image embedding ids
    :param embs: the concatenated image embeddings
    :param dirname: directory that the files will be saved to
    :param vid: view id of the images
    :return: None
    """
    emb_path = f'{dirname}/abc_clip_emb_{vid}.pt'
    print(embs.shape)
    print(f'Saving embeddings to ' + emb_path)
    torch.save(embs, emb_path)

    id_path = f'{dirname}/abc_emb_idx_{vid}.json'
    print(len(ids))
    print(f'Saving embeddings to ' + id_path)
    with open(id_path, 'w', encoding='utf8') as fp:
        json.dump(ids, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='png_txt_small/',
                        help='Path to the directory that saves part images')
    parser.add_argument('--vid', type=str, default='01',
                        help='The view id')
    args = parser.parse_args()

    vid = args.vid
    image_folder = args.image_folder
    image_list = sorted(glob.glob(image_folder + '*/*.png'))  # 10152, in alphabetical order, to maintain order in the metadata

    body_id_list = get_body_id(image_list)  # 423

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    clip, vis_processors, txt_processors = load_model_and_preprocess(name='clip_feature_extractor', model_type="base",
                                                                     is_eval=True, device=device)

    # Instantiate the dataset and the dataloader
    images = ABCPartImageDataset(body_id_list, vid, vis_processors, txt_processors, device)
    imageloader = data.DataLoader(images, shuffle=False, batch_size=64)

    image_ids, image_embs = get_emb(imageloader, clip)
    save_emb(image_ids, image_embs, 'processed_data', vid)

