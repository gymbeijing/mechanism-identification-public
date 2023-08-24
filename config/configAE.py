import argparse


class ConfigAE(object):
	def __init__(self, phase):
		self.is_train = phase == "train"
		parser, args = self.parse()
		self.parser = parser
		self.args = args

		self.set_configuration()
		
	def set_configuration(self):
		self.dim_emb = 512 * (self.args.n_neighbour + 1)
		self.dim_z = 512

	@staticmethod
	def parse():
		parser = argparse.ArgumentParser()

		parser.add_argument('--batch_size', type=int, default=64, help="batch size")
		parser.add_argument('--max_epochs', type=int, default=100, help="total number of training epochs")
		parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
		parser.add_argument('--emb_dir', type=str, default="../processed_data", help="directory to the saved embeddings")
		parser.add_argument('--n_neighbour', type=int, default=9, help="number of neighbouring parts for each central part")
		####### of no use, just a temporary fix for the bash script running problem
		parser.add_argument('--mse_lambda', type=int, default=1, help="weights for the mse loss")
		parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')

		args = parser.parse_args()
		return parser, args
