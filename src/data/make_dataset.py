# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from numpy import load
from torch.utils.data import TensorDataset

"""
    RUN: python make_dataset.py ../../data/raw/ ../../data/processed/
    RUN: black data/make_dataset.py to clean format dataset
"""


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    test_set = load(input_filepath + "test.npz")
    train0 = load(input_filepath + "train_0.npz")
    train1 = load(input_filepath + "train_1.npz")
    train2 = load(input_filepath + "train_2.npz")
    train3 = load(input_filepath + "train_3.npz")
    train4 = load(input_filepath + "train_4.npz")
    argsImgs = (
        train0["images"],
        train1["images"],
        train2["images"],
        train3["images"],
        train4["images"],
    )
    argsLabels = (
        train0["labels"],
        train1["labels"],
        train2["labels"],
        train3["labels"],
        train4["labels"],
    )
    train_set_img = np.concatenate(argsImgs)
    train_set_lb = np.concatenate(argsLabels)
    print(train_set_img)
    # print(torch.Tensor(train_set_img))
    train_dataset = TensorDataset(
        torch.Tensor(train_set_img), torch.Tensor(train_set_lb)
    )
    test_dataset = TensorDataset(
        torch.Tensor(test_set["images"]), torch.Tensor(test_set["labels"])
    )

    # Save data to file
    torch.save(train_dataset, output_filepath + "train.pt")
    torch.save(test_dataset, output_filepath + "test.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # RUN: python make_dataset.py ../../data/raw/ ../../data/processed/
    main()
