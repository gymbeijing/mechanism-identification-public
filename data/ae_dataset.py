from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import json
import os
import logging
from scipy.ndimage import shift


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def get_dataloader_from_tensor(tensor, config, shuffle=None):
	is_shuffle = False if shuffle is None else True
	my_dataset = TensorDataset(tensor)
	my_dataloader = DataLoader(my_dataset, batch_size=config.args.batch_size, shuffle=is_shuffle)

	return my_dataloader


def get_dataloader(phase, config, shuffle=None):
	is_shuffle = phase=='train' if shuffle is None else shuffle

	dt = AEDataset(phase)
	# is_shuffle = False
	dataloader = DataLoader(dt, batch_size=config.args.batch_size, shuffle=is_shuffle)
	return dataloader


class AEDataset(Dataset):
	#def __init__(self, config):
	def __init__(self, phase="train"):
		super(AEDataset, self).__init__()

		# Set the phase
		self.phase = phase
		train_test_path = os.path.join("./raw_data", "train_test.json")
		with open(train_test_path, 'r') as fp:
			train_test = json.load(fp)

		# All the embeddings
		#all_emb_path = os.path.join(config.emb_dir, "mean_pooled_emb.pt")
		all_emb_path = os.path.join("./processed_data", "mean_pooled_emb.pt")
		self.all_emb = torch.load(all_emb_path)   # [141245, 512]
		pad = torch.zeros(1, 512)
		self.all_emb = torch.cat((self.all_emb, pad), 0)   # [141246, 512]

		# All the metadata
		#all_md_path = os.path.join(config.emb_dir, "emb_idx_filtered.json")
		all_md_path = os.path.join("./processed_data", "emb_idx_filtered.json")
		with open(all_md_path, 'r') as fp:
			self.all_md = json.load(fp)   # .shape[0]=141245
		self.all_md.append("pad")   # .shape[0]=141246

		#n_neighbour = config.n_neighbour
		n_neighbour = 9

		# All the preprocessed part graphs
		#all_part_graph_path = os.path.join(config.emb_dir, "part_graphs_dim=5.json")
		all_part_graph_path = os.path.join("./processed_data", "part_graphs_dim=5.json")
		with open(all_part_graph_path, 'r') as fp:
			self.all_part_graph = json.load(fp)

		# Number of part graphs in total, train: 42,919, test: 10,377
		n_part_graph = 0
		for a_id, a_graphs in self.all_part_graph.items():
			if a_id in train_test[phase]:
				n_part_graph += len(a_graphs)

		# Map part name to its idx in the metadata/embeddings
		part_name_idx_map = dict()
		for idx, p_name in enumerate(self.all_md):
			part_name_idx_map[p_name] = idx

		#logging.info(f'pad\'s corresponding index is {part_name_idx_map["pad"]}')

		#self.all_data = torch.ones((n_part_graph, n_neighbour+1), dtype=torch.long)*part_name_idx_map["pad"]   # initialize to be all pad's index
		self.all_data = torch.full((n_part_graph, n_neighbour+1), fill_value=part_name_idx_map["pad"], dtype=torch.long)   # initialize to be all pad's index

		# Fill in the matrix
		r = 0
		for a_id, a_graphs in self.all_part_graph.items():
			if a_id in train_test[phase]:
				for c_part, neigh_parts in a_graphs.items():
					self.all_data[r][0] = part_name_idx_map[c_part]
					c = 1
					for neigh_part in neigh_parts:
						if c <= n_neighbour:   # only keep the first n_neighbour neighbouring parts
							self.all_data[r][c] = part_name_idx_map[neigh_part]
						else:
							break
						c += 1
					r += 1

		self.all_data = self.all_data   # ?

		'''
		Add mask
		'''
		pad_idx = self.all_emb.shape[0] - 1
		self.mask = (self.all_data < pad_idx).long()   # [n_part_graphs, 10]
		self.mask = self.mask.roll(1, 1)   # shift by 1 along axis 1
		# [n_part_graphs, 10], namely [42,919, 10] for the training data,
		# and [10,377, 10] for the testing data, element being either 0 or 1
		self.mask[:, 0] = 1   # set the first [pad]'s mask to 1, if there is any [pad]

	def __len__(self):
		return self.all_data.shape[0]

	def __getitem__(self, idx):
		# emb_list = []
		indices = self.all_data[idx]
		#indices = torch.LongTensor(indices)
		#for ind in indices:
		#	emb_list.append(self.all_emb[int(ind)])
		#item = torch.cat(emb_list, 0)   # [5120]
		item = self.all_emb[indices].flatten()
		mask = self.mask[idx].repeat_interleave(512).flatten()   # [10] -> [5120]

		return item, mask


if __name__ == "__main__":
	dataset = AEDataset("test")
	item, mask = dataset[0]
	print(item[-522:-502])
	print(item.shape)
	print(mask)
	print(mask.shape)
	print(len(dataset))
