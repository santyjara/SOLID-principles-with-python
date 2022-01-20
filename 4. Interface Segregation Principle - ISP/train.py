import argparse
from typing import Dict, List, Tuple, Union

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from base import Model
from rnn_model import RNNModel
from sentence_encoder_model import SentenceEncoderModel


def make_dataset(config_data: Dict[str, str]) -> Tuple[List[str], List[int]]:
    """
    Build dataset based on a configuration file

    Parameters
    ----------
    config_data : Dict[str, str]
        Path to configuration file

    Returns
    -------
    Tuple[List[str], List[int]]
        Sentences, labels
    """
    data_path, sentence_column, target_column = (
        config_data["data_path"],
        config_data["sentence_column"],
        config_data["target_column"],
    )
    data = pd.read_csv(data_path)
    sentences = data[sentence_column].to_list()
    labels = data[target_column].apply(lambda x: 1 if x == "positive" else 0).to_list()
    return sentences, labels


def read_yaml(yaml_path: str) -> Dict[str, Union[str, int]]:
    """
    Read .yaml file

    Parameters
    ----------
    yaml_path : str
        Path to yaml file

    Returns
    -------
    Dict[str, Union[str, int]]
        Yaml file content
    """
    with open(yaml_path) as f:
        my_dict = yaml.safe_load(f)
    return my_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Enter the mode you want to use",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--model_type",
        choices=["RNN", "sentence-encoder"],
        default="RNN",
        help="Enter the model you want to use",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pretrained_model",
        help="Path of the pretrained model if exists",
    )
    args = parser.parse_args()
    strategy_mapping = {"RNN": RNNModel, "sentence-encoder": SentenceEncoderModel}
    config = read_yaml("config.yml")

    sentences, labels = make_dataset(config["data"])
    x_train, x_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    # Pass by composition the strategy you want to use into the context
    model = Model(strategy=strategy_mapping[args.model_type](args.pretrained_model))

    if args.mode == "train":
        model.train(x_train, y_train, x_test, y_test, config[args.model_type])
    else:
        if not args.pretrained_model:
            raise FileNotFoundError(
                "Enter a valid model path to be evaluated, use the tag -p"
            )
        model.eval(x_test, y_test)
