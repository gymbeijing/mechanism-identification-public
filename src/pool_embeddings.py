import argparse
import os
import glob
import json
import torch


def load_metadata(metadata_filepaths):
	metadata_dict = dict()
	for metadata_filepath in metadata_filepaths:
		vid = metadata_filepath.split('/')[-1].split('.')[0].split('_')[-1]
		with open(metadata_filepath, 'r') as f:
			d = json.load(f)
			metadata_dict[vid] = ['_'.join(name.split('_')[:-1]) for name in d]
		print(f'{vid}: {len(metadata_dict[vid])}')

	return metadata_dict


def compute_intersect(metadata_dict):
	metadata_list = list(metadata_dict.values())
	intersect = set.intersection(*map(set, metadata_list))
	print(f'length of intersection: {len(intersect)}')
	return intersect


def load_emb(emb_filepaths):
	emb_dict = dict()
	for emb_filepath in emb_filepaths:
		vid = emb_filepath.split('/')[-1].split('.')[0].split('_')[-1]
		with open(emb_filepath, 'r') as f:
			emb_dict[vid] = torch.load(emb_filepath)
		print(f'{vid}: {emb_dict[vid].shape}')
	
	return emb_dict
		


if __name__ == '__main__':
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--embs_folder', type=str, help='Path to the folder that saves all the embeddings')
	parser.add_argument('--pooling_strategy', type=str, help='Pooling strategy to be used')
	args = parser.parse_args()

	# Load in all the image embeddings
	emb_filepaths = glob.glob(f'{args.embs_folder}/*.pt')
	metadata_filepaths = glob.glob(f'{args.embs_folder}/*.json')
	#print(emb_filepaths)
	#print(metadata_filepaths)

	# Load embedding metadata into a dictionary
	metadata_dict = load_metadata(metadata_filepaths)
	
	# Compute the intersect parts name
	intersect = compute_intersect(metadata_dict)   # 141245
	
	emb_dict = load_emb(emb_filepaths)
