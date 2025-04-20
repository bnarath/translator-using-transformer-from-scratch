from typing import List


def is_utf8_encodable(example):
    """Function to check UTF-8 validity"""
    try:
        example["src"].encode("utf-8")
        example["tgt"].encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def is_len_valid(tokens: List, max_seq_len: int):
    "Check if total len of tokens are within the limit"
    return True if len(tokens) <= max_seq_len else False
