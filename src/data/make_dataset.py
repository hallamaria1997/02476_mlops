# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # Loading Raw data
    train_df = pd.read_csv(os.path.join(input_filepath,"train.csv"))
    test_df = pd.read_csv(os.path.join(input_filepath,"test.csv"))
    # Dropping rows with missing data
    train_df.dropna()
    test_df.dropna()
    # Removing unused data column
    train_df.drop(columns="selected_text", inplace=True)
    # Making sure all text data has same datatype
    train_df.text = train_df.text.apply(lambda x: str(x))
    test_df.text = test_df.text.apply(lambda x: str(x))
    # Removing empty spaces and quotation marks at beginning and end of text
    train_df.text = train_df.text.apply(lambda x: x.strip())
    train_df.text = train_df.text.apply(lambda x: x.strip('\"'))
    train_df.text = train_df.text.apply(lambda x: x.strip('\''))
    test_df.text = test_df.text.apply(lambda x: x.strip())
    test_df.text = test_df.text.apply(lambda x: x.strip('\"'))
    test_df.text = test_df.text.apply(lambda x: x.strip('\''))
    # Saving processed data
    train_df[["textID", "text", "sentiment"]].to_csv(os.path.join(output_filepath,
                                                                "train.csv"),
                                                    index=False)
    test_df[["textID", "text", "sentiment"]].to_csv(os.path.join(output_filepath,
                                                                "test.csv"),
                                                    index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
