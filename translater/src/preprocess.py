import torch
import torch.nn as nn
from data.retrieve import Retriever
from utils.process import is_utf8_encodable

from config.data_dictionary import ROOT, Train, HuggingFaceData, BPE_Enum
from utils.utils import get_bpe_path

from collections import defaultdict

import pickle
import logging
from typing import Literal, List, Dict
from datasets.arrow_dataset import Dataset


class Preprocessor:
    """
    - Download the data from HuggingFace
    - Split data as Train/Validation/Test
    - Save all the files - raw data (Train/Validation/Test)
    - Discard sentence pairs if there is any error on UTF-8 conversion
    - Byte Pair Encoding Training (BPE) on training data to generate a vocabulary size of 500 tokens
    - Save the BPE mapping
    - Have a logic to convert sentences into BPE tokens
    - Have a logic to convert BPE tokens back to sentences
    """

    def process(self):
        logging.info("1. Retrive data from hugging face")
        retriever = Retriever()
        logging.info(retriever.train_data)
        logging.info(retriever.val_data)
        logging.info(retriever.test_data)

        logging.info("2. Getting only valid sentence pairs")

        self.valid_train_data = retriever.train_data.filter(is_utf8_encodable)
        self.valid_val_data = retriever.val_data.filter(is_utf8_encodable)
        self.valid_test_data = retriever.test_data.filter(is_utf8_encodable)
        logging.info("After discarding invalid pairs")
        logging.info(self.valid_train_data)
        logging.info(self.valid_val_data)
        logging.info(self.valid_test_data)

        logging.info("3. Training & Saving BPE")
        corpus = create_corpus(self.valid_train_data.select(range(10)))
        bpe = BPE(corpus)
        bpe.train()
        bpe.save_artifacts()
        logging.info(f"Encoding of 'Hello World!': {bpe.encode('Hello World!')}")
        logging.info(f"Decoding back: {bpe.decode(bpe.encode('Hello World!'))}")


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

        for _ in range(self.final_vocab_size - self.initial_vocab_size):
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
            for i in range(len(tokens) - 1):
                if (tokens[i], tokens[i + 1]) == most_freq_pair:
                    new_tokens.append(self.map[most_freq_pair])
                else:
                    new_tokens.append(tokens[i])
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
        return b"".join(tokens).decode("utf-8")

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


class BatchTokenizer(nn.Module):
    def __init__(
        self,
        max_seq_length,
        vocab_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
        UNKNOWN_TOKEN,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.vocab_to_index = vocab_to_index
        self.vocab_size = len(vocab_to_index)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.UNKNOWN_TOKEN = UNKNOWN_TOKEN
        self.special_tokens = {
            self.START_TOKEN,
            self.END_TOKEN,
            self.PADDING_TOKEN,
            self.UNKNOWN_TOKEN,
        }

    def forward(
        self,
        batch: List[str],
        start_token: bool = False,
        end_token: bool = False,
    ) -> torch.Tensor:  # (batch, max_sequence_len)
        """Tokenize in batches"""
        tokenized_batch = []
        for sentence in batch:
            tokenized_batch.append(
                self.tokenize(
                    sentence,
                    self.vocab_to_index,
                    self.max_seq_length,
                    start_token,
                    end_token,
                )
            )
        tokenized_batch = torch.stack(tokenized_batch)
        return tokenized_batch

    def tokenize(
        self,
        sentence: List[str],
        vocab_to_id: Dict[str, int],
        max_seq_len: int,
        start_token: bool = False,
        end_token: bool = False,
    ) -> List[int]:
        """Tokenize a sentence. Optionally add start and end tokens. Always pad with padding token."""
        tokens = []
        if start_token:
            tokens = [vocab_to_id[self.START_TOKEN]]

        i = 0
        while i < len(sentence):
            matched = False
            # check if sentence at i starts with special token
            for special_token in self.special_tokens:
                if sentence[i].startswith(special_token):
                    # In case of inference, as self.PADDING_TOKEN is part of vocab, output becomes self.PADDING_TOKEN.
                    # As we use the same output as input, self.PADDING_TOKEN comes as a normal token. In such cases, take it as self.UNKNOWN_TOKEN
                    id_of_special_token = (
                        vocab_to_id[special_token]
                        if special_token != self.PADDING_TOKEN
                        else vocab_to_id[self.UNKNOWN_TOKEN]
                    )
                    tokens.append(id_of_special_token)
                    i += len(special_token)
                    matched = True
                    break

            if not matched:  # If no special token matched, tokenize character-wise
                tokens.append(
                    vocab_to_id.get(sentence[i], vocab_to_id[self.UNKNOWN_TOKEN])
                )
                i += 1

        if end_token:
            tokens.append(vocab_to_id[self.END_TOKEN])

        for _ in range(len(tokens), max_seq_len):
            tokens.append(vocab_to_id[self.PADDING_TOKEN])

        # Ensure max sequence length constraint
        tokens = tokens[:max_seq_len]

        return torch.tensor(tokens, dtype=torch.long)


class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        max_seq_length,
        d_model,
        vocab_to_index,
        drop_prob,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model  # Embedding dimension
        self.vocab_to_index = vocab_to_index
        self.drop_prob = drop_prob  # Drop after embedding + pos encoding
        self.PADDING_TOKEN = PADDING_TOKEN
        self.vocab_size = len(vocab_to_index)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=vocab_to_index[self.PADDING_TOKEN],
        )
        self.positional_encoder = PositionalEncoder(
            self.max_seq_length, self.d_model
        )  # TBD
        self.dropout = nn.Dropout(0.1)
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
