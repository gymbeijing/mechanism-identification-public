import argparse


class ConfigGAN(object):
    def __init__(self, phase):
        self.is_train = phase == "train"
        parser, args = self.parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.n_dim = 64
        self.h_dim = 512
        self.z_dim = 512

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch_size', type=int, default=64, help="batch size")
        parser.add_argument('--max_epochs', type=int, default=100, help="total number of training epochs")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--z_dir', type=str, default="../model_outputs",
                            help="directory to the saved latent variables")

        args = parser.parse_args()
        return parser, args
