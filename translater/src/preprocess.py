import torch
import torch.nn as nn
from data.retrieve import Retriever

from config.data_dictionary import HuggingFaceData, BPE_Enum
from utils.utils import get_bpe_path, get_device
from utils.process import is_len_valid

from collections import defaultdict

import pickle
import logging
from tqdm import tqdm
from typing import List, Tuple
from datasets.arrow_dataset import Dataset


class Preprocessor:
    """
    This class is used for BPE training
    - Download the data from HuggingFace
    - Split data as Train/Validation/Test
    - Discard sentence pairs if there is any error on UTF-8 conversion
    - Save all the files - raw valid data (Train/Validation/Test)
    - Byte Pair Encoding Training (BPE) on training data to generate a vocabulary size of 500 tokens
    - Save the BPE mapping
    - Have a logic to convert sentences into BPE tokens
    - Have a logic to convert BPE tokens back to sentences
    """

    def process(self):
        logging.info(
            "1. Retrive data from hugging face and get only the valid pairs (utf-8 encodable)"
        )
        retriever = Retriever()
        logging.info(retriever.train_data)
        logging.info(retriever.val_data)
        logging.info(retriever.test_data)

        logging.info("3. Training & Saving BPE")
        corpus = create_corpus(retriever.train_data)
        bpe = BPE(corpus)
        bpe.train()
        bpe.save_artifacts()
        logging.info(f"Encoding of 'Hello World!': {bpe.encode('Hello World!')}")
        logging.info(f"Decoding back: {bpe.decode(bpe.encode('Hello World!'))}")


class Processor:
    """
    This class is used before translator training
    - Retrieve the Train/Validation/Test data
    - Encode using BPE encoder
    - Discard if len > max_seq_length in case of src and len > max_seq_length - 1 in case of target(END token to be added in the labels)
    """

    def __init__(self):
        # placeholder
        self.valid_train_token_pairs = None
        self.valid_val_token_pairs = None
        self.valid_test_token_pairs = None

    def process(self, bpe: "BPE", to_print: bool = False):
        logging.info("1. Retrieves data train / val /  test data")
        retriever = Retriever()
        self.valid_train_token_pairs = self.get_valid_token_pairs(
            dataset=retriever.train_data,
            max_seq_len=HuggingFaceData.max_length.value,
            total=HuggingFaceData.max_train_size.value,
            bpe=bpe,
            to_print=to_print,
        )
        self.valid_val_token_pairs = self.get_valid_token_pairs(
            dataset=retriever.val_data,
            max_seq_len=HuggingFaceData.max_length.value,
            total=HuggingFaceData.max_val_size.value,
            bpe=bpe,
            to_print=to_print,
        )
        self.valid_test_token_pairs = self.get_valid_token_pairs(
            dataset=retriever.test_data,
            max_seq_len=HuggingFaceData.max_length.value,
            total=HuggingFaceData.max_test_size.value,
            bpe=bpe,
            to_print=to_print,
        )
        logging.info(f"Train data len : {len(self.valid_train_token_pairs)}")
        logging.info(f"Val data len : {len(self.valid_val_token_pairs)}")
        logging.info(f"Test data len : {len(self.valid_test_token_pairs)}")
        return (
            self.valid_train_token_pairs,
            self.valid_val_token_pairs,
            self.valid_test_token_pairs,
        )

    def get_valid_token_pairs(
        self,
        dataset: Dataset,
        total: int,
        max_seq_len: int,
        bpe: "BPE",
        to_print: bool = False,
    ) -> List[Tuple[List[int], List[int]]]:
        """Given Hugging face dataset, return valid token list pairs."""
        valid_token_pairs = []
        length_exceeded_count = 0

        for data in tqdm(dataset.select(range(total))):
            src_tokens = bpe.encode(data["src"])
            tgt_tokens = bpe.encode(data["tgt"])

            src_len_check = is_len_valid(data["src"], max_seq_len=max_seq_len)
            tgt_len_check = is_len_valid(
                data["tgt"], max_seq_len=max_seq_len - 1
            )  # To account for start/end

            if src_len_check and tgt_len_check:
                valid_token_pairs.append((src_tokens, tgt_tokens))
            else:
                length_exceeded_count += 1

        if to_print:
            logging.info(
                f"{length_exceeded_count} no of examples found exceeding max seq length of {max_seq_len} out of total {total} examples."
            )

        return valid_token_pairs


class BPE:
    """A class to train and encode Byte Pair Encoding from corpus (which is tokenized)"""

    def __init__(
        self, corpus: List[List[int]], final_vocab_size: int = BPE_Enum.vocab_size.value
    ):
        self.corpus = corpus
        self.final_vocab_size = final_vocab_size
        self.initial_vocab_size = 256
        self.vocab_size = self.initial_vocab_size
        self.map = {}  # (pair -> token)
        self.reverse_map = {}  # (token -> bytes)

    def train(self):

        for _ in tqdm(range(self.final_vocab_size - self.initial_vocab_size)):
            most_freq_pair = self.get_most_freq_pair()
            if most_freq_pair is None:
                break
            new_token_id = self.vocab_size
            self.vocab_size += 1
            self.map[most_freq_pair] = new_token_id
            self.corpus = self.update_corpus(most_freq_pair, self.corpus)

        # ensure sorted order for the maps
        self.map = dict(sorted(self.map.items(), key=lambda x: x[1]))

        # generate a reverse_map that maps a token to bytes
        vocab = {i: bytes([i]) for i in range(self.initial_vocab_size)}
        for (l, r), token in self.map.items():
            vocab[token] = vocab[l] + vocab[r]
        self.reverse_map = vocab

    def get_counts_of_subsequent_pairs(self, corpus: List[List[int]]):
        counts = defaultdict(int)
        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] += 1
        return counts

    def get_most_freq_pair(self):
        pair_counts = self.get_counts_of_subsequent_pairs(self.corpus)
        if not pair_counts:
            return None
        return max(pair_counts, key=pair_counts.get)

    def update_corpus(self, most_freq_pair, corpus: List[List[int]]):
        new_corpus = []
        for tokens in corpus:
            new_tokens = []
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == most_freq_pair:
                    new_tokens.append(self.map[most_freq_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if len(tokens) > 1 and (tokens[-2], tokens[-1]) != most_freq_pair:
                new_tokens.append(tokens[-1])
            new_corpus.append(new_tokens)
        return new_corpus

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        corpus = [tokens]
        while True:
            counts = self.get_counts_of_subsequent_pairs(corpus)
            if not counts:
                break
            pair = min(
                counts, key=lambda x: self.map.get(x, float("inf"))
            )  # get the earlier pair that was merged. Earliest pair will have the lowest value (We start with 256)
            if pair not in self.map:
                break
            # else
            corpus = self.update_corpus(pair, corpus)
        return corpus[0]

    def decode(self, tokens: List[int]) -> str:
        tokens = [self.reverse_map[token] for token in tokens]
        return b"".join(tokens).decode("utf-8", errors="replace")

    def save_artifacts(self):
        artifacts = {
            "vocab_size": self.vocab_size,
            "map": self.map,
            "reverse_map": self.reverse_map,
        }
        file_uri = get_bpe_path()
        with open(file_uri, "wb") as f:
            pickle.dump(artifacts, f)

    def load_artifacts(self):
        file_uri = get_bpe_path()
        with open(file_uri, "rb") as f:
            artifacts = pickle.load(f)
            self.vocab_size = artifacts["vocab_size"]
            self.map = artifacts["map"]
            self.reverse_map = artifacts["reverse_map"]


def create_corpus(data: Dataset) -> List[int]:
    """Create a corpus from the dataset where data is converted into unicode code points -> utf-8 encoded"""

    corpus = []
    for row in data:
        corpus.append(list(row["src"].encode("utf-8")))
        corpus.append(list(row["tgt"].encode("utf-8")))

    return corpus


class BatchPadder(nn.Module):
    def __init__(
        self,
        max_seq_length,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def forward(
        self,
        batch: List[List[int]],
        start_token: bool = False,
        end_token: bool = False,
    ) -> torch.Tensor:  # (batch, max_sequence_len)
        """Pad the sentence tokens in batches"""
        tokenized_batch = []
        for sentence_tokens in batch:
            tokenized_batch.append(
                self.pad(
                    sentence_tokens,
                    self.max_seq_length,
                    start_token,
                    end_token,
                )
            )
        tokenized_batch = torch.stack(tokenized_batch)
        return tokenized_batch

    def pad(
        self,
        tokenized_sentence: List[int],
        max_seq_len: int,
        start_token: bool = False,
        end_token: bool = False,
    ) -> List[int]:
        """Pad the sentence tokens upto the same max sequence length. Optionally add start and end tokens."""
        tokens = []
        if start_token:
            tokens = [BPE_Enum.special_tokens.value[self.START_TOKEN]]

        tokens.extend(tokenized_sentence)

        if end_token:
            tokens.append(BPE_Enum.special_tokens.value[self.END_TOKEN])

        for _ in range(len(tokens), max_seq_len):
            tokens.append(BPE_Enum.special_tokens.value[self.PADDING_TOKEN])

        # Ensure max sequence length constraint
        tokens = tokens[:max_seq_len]

        return torch.tensor(tokens, dtype=torch.long)


class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        max_seq_length,
        d_model,
        drop_prob,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model  # Embedding dimension
        self.drop_prob = drop_prob  # Drop after embedding + pos encoding
        self.PADDING_TOKEN = PADDING_TOKEN
        self.embedding = nn.Embedding(
            BPE_Enum.vocab_size.value + len(BPE_Enum.special_tokens.value),
            self.d_model,
            padding_idx=BPE_Enum.special_tokens.value[self.PADDING_TOKEN],
        )
        self.positional_encoder = PositionalEncoder(self.max_seq_length, self.d_model)
        self.dropout = nn.Dropout(0.1)
        self.device = get_device()
        self.to(self.device)

    def forward(self, x):
        # x: (batch, max_seq_length)
        x = self.embedding(x)  # (batch, max_seq_len, d_model)
        pos = self.positional_encoder().to(self.device)  # (batch, max_seq_len, d_model)
        x = x + pos
        x = self.dropout(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.PE = self.get_encoding()
        self.device = get_device()
        self.to(self.device)

    def forward(self):
        # x: (batch, max_seq_len, d_model)
        return self.PE

    def get_encoding(self):
        even_i = torch.arange(0, self.d_model, 2)
        denominator = torch.pow(10000, even_i / self.d_model)
        pos = torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_PE = torch.sin(pos / denominator)
        odd_PE = torch.cos(pos / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
