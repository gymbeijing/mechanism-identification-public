import argparse

class ConfigAE(object):
	def __init__(self, phase):
		self.is_train = phase == "train"

		self.set_configuration()

		parser, args = self.parse()
		
	def set_configuration(self):
		self.dim_emb = 512 * args.n_neighbour
		self.dim_z = 512

	def parse(self):
		parser = argparse.ArgumentParser()

		parser.add_argument('--batch_size', type=int, default=64, help="batch size")
		parser.add_argument('--epochs', type=int, default=10, help="total number of training epochs")
		parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
		parser.add_argument('--emb_dir', type=str, default="../processed_data", help="directory to the saved embeddings")
		parser.add_argument('--n_neighbour', type=int, default=10, help="number of neighbouring parts for each central part")

		args = parser.parse_args()
		return parser, args
