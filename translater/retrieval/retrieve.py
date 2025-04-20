"""Contains all the code base to retrieve training data"""

from datasets import load_dataset, load_from_disk
from utils.process import is_utf8_encodable
from config.data_dictionary import ROOT, HuggingFaceData
from pathlib import Path
import os
import logging


from utils.timing import timeit


class Retriever:

    def __init__(self):
        self.train_data_path = ROOT / Path(HuggingFaceData.save_train_location.value)
        self.val_data_path = ROOT / Path(HuggingFaceData.save_val_location.value)
        self.test_data_path = ROOT / Path(HuggingFaceData.save_test_location.value)
        self.train_data, self.val_data, self.test_data = self.retrieve_data()
        self.print_counts()

    def print_counts(self):
        logging.info(f"Train data length: {len(self.train_data)}")
        logging.info(f"Validation data length: {len(self.val_data)}")
        logging.info(f"Test data length: {len(self.test_data)}")

    @timeit
    def retrieve_data(self):
        directory_path = os.path.dirname(self.train_data_path)
        if not (
            self.train_data_path.exists()
            and self.test_data_path.exists()
            and self.val_data_path.exists()
        ):
            # Create the directory if it doesn't exist
            os.makedirs(directory_path, exist_ok=True)
            ds = load_dataset(
                HuggingFaceData.dataset.value,
                name=HuggingFaceData.name.value,
                split=HuggingFaceData.split.value,  # train
            )
            ds.remove_columns(HuggingFaceData.remove_feature.value)
            # First split into train and temp (val+test)
            train_test_split = ds.train_test_split(
                test_size=HuggingFaceData.test_ratio.value
                + HuggingFaceData.val_ratio.value,
                seed=HuggingFaceData.seed.value,
            )
            val_test_split = train_test_split["test"].train_test_split(
                test_size=HuggingFaceData.test_ratio.value
                / (HuggingFaceData.test_ratio.value + HuggingFaceData.val_ratio.value),
            )

            logging.info("Getting only valid sentence pairs")

            train_test_split["train"] = train_test_split["train"].filter(
                is_utf8_encodable
            )
            val_test_split["train"] = val_test_split["train"].filter(is_utf8_encodable)
            val_test_split["test"] = val_test_split["test"].filter(is_utf8_encodable)

            train_test_split["train"].save_to_disk(self.train_data_path)
            val_test_split["train"].save_to_disk(self.val_data_path)
            val_test_split["test"].save_to_disk(self.test_data_path)

            return (
                train_test_split["train"],
                val_test_split["train"],
                val_test_split["test"],
            )
        else:
            logging.info(f"Data already exists at {directory_path}")
            return (
                load_from_disk(self.train_data_path),
                load_from_disk(self.val_data_path),
                load_from_disk(self.test_data_path),
            )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    retriever = Retriever()
