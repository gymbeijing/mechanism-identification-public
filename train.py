from .config import ConfigAE
from dataset.ae_dataset import get_dataloader
from tqdm import tqdm


def main():
	cfg = ConfigAE('train')
	train_loader = get_dataloader('train', cfg)


if __name__ == '__main__':
	main()
