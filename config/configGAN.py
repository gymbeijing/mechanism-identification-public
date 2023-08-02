import argparse


class ConfigGAN(object):
    def __init__(self, phase):
        self.is_train = phase == "train"
        self.phase = phase
        parser, args = self.parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.n_dim = 64
        self.h_dim = 512
        self.z_dim = 512

        self.critic_iters = 5
        self.gp_lambda = 5

    # @staticmethod
    def parse(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        parser.add_argument('--max_epochs', type=int, default=500, help="total number of training epochs")
        parser.add_argument('--lr', type=float, default=2e-4, help="initial learning rate")
        parser.add_argument('--z_file', type=str, default=f"./model_outputs/z_{self.phase}_0717_170857.pt",
                            help="directory to the saved latent variables")
        parser.add_argument('--c_emb_file', type=str, default=f"./processed_data/center_emb_{self.phase}.pt",
                            help="directory to the saved latent variables")
        parser.add_argument('--mse_lambda', type=int, default=1, help="weights for the mse loss")

        args = parser.parse_args()
        return parser, args
