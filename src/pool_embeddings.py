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
		#print(f'{vid}: {len(metadata_dict[vid])}')

	return metadata_dict


def compute_intersect(metadata_dict):
	metadata_list = list(metadata_dict.values())
	intersect = set.intersection(*map(set, metadata_list))
	#print(f'length of intersection: {len(intersect)}')
	return intersect


def load_emb(emb_filepaths):
	emb_dict = dict()
	for emb_filepath in emb_filepaths:
		vid = emb_filepath.split('/')[-1].split('.')[0].split('_')[-1]
		with open(emb_filepath, 'r') as f:
			emb_dict[vid] = torch.load(emb_filepath)
		#print(f'{vid}: {emb_dict[vid].shape}')
	
	return emb_dict


def filter_emb(emb_dict, intersect, metadata_dict):
	emb_filtered_dict = dict()
	for vid, emb in emb_dict.items():
		m_data = metadata_dict[vid]
		keep_indice = []
		for idx, d in enumerate(m_data):
			if d in intersect:
				keep_indice.append(idx)
		emb_dict[vid] = emb[keep_indice]   # warning: in-place operation, better be emb_filtered_dict[vid] = emb[keep_indice], and later return emb_filtered_dict, new_metadata
		#print(f'{vid}: {emb_dict[vid].shape}')   # [141245, 512]
		if vid == '00':   # only use 00 to find the metadata.It will be the same whichever {vid} folder to use
			new_metadata = [metadata_dict[vid][idx] for idx in keep_indice]

	return emb_dict, new_metadata
		

def pool_embedding(emb_dict, strategy='mean'):
	embs = list(emb_dict.values())
	stacked_emb = torch.stack(embs, 0)   # [24, 141245, 512]

	if strategy == 'mean':
		pooled_emb = torch.mean(stacked_emb, dim=0)
	if strategy == 'max':
		pooled_emb = torch.max(stacked_emb, dim=0).values

	return pooled_emb


def save_emb_to_file(emb, metadata, args):
	print(f'Saving {args.pooling_strategy} pooled embeddings to {args.embs_folder}/{args.pooling_strategy}_pooled_emb.pt')
	torch.save(emb, f'{args.embs_folder}/{args.pooling_strategy}_pooled_emb.pt')
	print(f'Saving filtered embedding metadata to {args.embs_folder}/emb_idx_filtered.json')
	with open(f'{args.embs_folder}/emb_idx_filtered.json', 'w') as fw:
		json.dump(metadata, fw, indent=4, ensure_ascii=False, sort_keys=False)

	return


if __name__ == '__main__':
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--embs_folder', type=str, default='../emb', help='Path to the folder that stores all the embeddings')
	parser.add_argument('--pooling_strategy', type=str, default='mean', help='Pooling strategy to be used')
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
	
	# Load embedding into a dictionary
	emb_dict = load_emb(emb_filepaths)

	# Filter out the part that doesn't have 24 views
	emb_dict, new_metadata = filter_emb(emb_dict, intersect, metadata_dict)

	# Pool embeddings according to the pooling strategy
	pooled_emb = pool_embedding(emb_dict, args.pooling_strategy)   # [141245, 512]

	save_emb_to_file(pooled_emb, new_metadata, args)
