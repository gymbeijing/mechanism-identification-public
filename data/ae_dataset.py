from torch.utils.data import Dataset, DataLoader
import torch
import json
import os


class AEDataset(Dataset):
	def __init__(self, config):
		super(AEDataset, self).__init__()
	
		# All the embeddings
		all_emb_path = os.path.join(config.emb_dir, "mean_pooled_emb.pt")
		self.all_emb = torch.load(all_emb_path)   # [141245, 512]
		pad = torch.zeros(1, 512)
		self.all_emb = torch.cat((self.all_emb, pad), 0) # [141246, 512]
		
		# All the metadata
		all_md_path = os.path.join(config.emb_dir, "emb_idx_filtered.json")
		with open(all_md_path, 'r') as fp:
			self.all_md = json.load(fp)   # 141245
		self.all_md["pad"] = self.all_emb.shape[0]-1   # 141246

		n_neighbour = config.n_neighbour

		# All the preprocessed part graphs
		all_part_graph_path = os.path.join(config.emb_dir, "part_graphs_dim=5.json")
		with open(all_part_graph_path, 'r') as fp:
			self.all_part_graph = json.load(fp)
		
		# Number of part graphs in total, ~53000
		n_part_graph = 0
		for a_id, a_graphs in self.all_part_graph.items():
			n_part_graph += len(a_graphs)
		
		# Map part name to its idx in the metadata/embeddings
		self.prt_name_idx_map = dict()
		for idx, p_name in enumerate(self.all_md):
			prt_name_idx_map[p_name] = idx

		self.all_data = torch.ones(n_part_graph, n_neighbour+1)*self.prt_name_idx["pad"]   # initialize to be all pad's index 

		r = 0
		for a_id, a_graphs in self.all_part_graph.items():
			for c_part, neigh_parts in a_graphs.items():
				self.all_data[r][0] = self.prt_name_idx_map[c_part]
				c = 1
				for neigh_part in neigh_parts:
					if c <= n_neighbour:   # only keep the first n_neighbour neighbouring parts
						self.all_data[r][c] = self.prt_name_idx_map[neigh_part]
					else:
						break
					c += 1
			r += 1

	def __len__(self):
		return self.all_data.shape[0]

	def __getitem__(self, idx):
		return self.all_data[idx]   # [512]
