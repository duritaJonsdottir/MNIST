import argparse
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from model import MyAwesomeModel
from torch.utils.data import DataLoader, TensorDataset


@click.command()
@click.argument("pretrained_model", type=click.Path(exists=True))
@click.argument("images_forprediction", type=click.Path())


# Should take a pretrained model and then make predictions on images
def main(pretrained_model, images_forprediction):

    """ images_forprediction should be a pickle file which i have not implemented yet. 
        Then the file can be run as :  
            python src/models/predict_model.py \
            models/my_trained_model.pt \ 
            data/example_images.npy 
    """
    # Initialize model
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(pretrained_model.load_model_from))
    model.eval()

    for images in images_forprediction:
        images = images[None, :, :, :]
        outputs = model(images)
        _, predicted = torch.max(outputs[0].data, 1)

    return predicted


if __name__ == "__main__":
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    # RUN: python make_dataset.py ../../data/raw/ ../../data/processed/
    main()
