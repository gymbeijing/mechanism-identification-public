import torch
from torch.utils.data import Dataset
from lavis.models import load_model_and_preprocess
import argparse
import glob
from PIL import Image
import torch.utils.data as data


class PartImageDataset(Dataset):
    def __init__(self, image_folder, model, vis_processors, txt_processors, device):
        self.image_folder = image_folder
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.image_list = glob.glob(self.image_folder + '/*.png')[:64]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        raw_image = Image.open(self.image_list[idx])
        caption = ""
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)
        sample = {"image": image, "text_input": [text_input]}

        features = model.extract_features(sample)
        features_image = features.image_embeds

        return features_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, help='Path to the directory that saves part images')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model, vis_processors, txt_processors = load_model_and_preprocess(name='clip_feature_extractor', model_type="base",
                                                                      is_eval=True, device=device)
    # print(vis_processors)

    images = PartImageDataset(args.image_folder, model, vis_processors, txt_processors, device)
    imageloader = data.DataLoader(images, shuffle=False, batch_size=32)

    batch_emb_list = []
    for i, batch_emb in enumerate(imageloader, 0):
        batch_emb_list.append(batch_emb)

    total_emb = torch.cat(batch_emb_list, dim=0)
    print(total_emb.shape)
