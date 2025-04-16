from config.data_dictionary import HuggingFaceData
from typing import List


def is_utf8_encodable(example):
    """Function to check UTF-8 validity"""
    try:
        example["src"].encode("utf-8")
        example["tgt"].encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False
