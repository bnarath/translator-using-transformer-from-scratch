"""
This is the dictionary containing parameter info for the package run
"""

from enum import Enum
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
NEG_INFINITY = -1e20


class HuggingFaceData(Enum):
    dataset = "ai4bharat/samanantar"
    name = "ml"
    split = "train"
    remove_feature = ["idx"]
    save_train_location = "data/english_malayalam_train.arrow"
    save_val_location = "data/english_malayalam_val.arrow"
    save_test_location = "data/english_malayalam_test.arrow"
    src_column = "src"
    tgt_column = "tgt"
    train_ratio = 0.5
    val_ratio = 0.25
    test_ratio = 0.25
    max_length = 300
    seed = 1
    max_train_size = 1000000
    max_val_size = 100000
    max_test_size = 100000


class BPE_Enum(Enum):
    vocab_size = 500
    bpe_file = "result/bpe.pkl"  # contains vocab_size, map, reverse_map as dict


class Train(Enum):
    batch_size = 64
    seed = 1
    learning_rate = 1e-4
    num_epochs = 10
    checkpoint_dir = "result/checkpoints"
    log_dir = "result/logs"


class Encoder_Enum(Enum):
    num_layers = 2
    d_model = (
        512  # the dimensionality of the model's hidden states or embeddings, q, k, v
    )
    # q_k_v_dim = 64 is deduced as d_modelnum_attention_heads as 8*64 = 512
    num_attention_heads = 8  # For self attention in both Encoder
    drop_prob = 0.1  # drop probability (10% dropout), happens after every layer norm and inside FFW
    hidden_dim = 2048  # dim of FFW nw's hidden layer


class Decoder_Enum(Enum):
    num_layers = 2
    d_model = 512
    num_attention_heads = 8
    drop_prob = 0.1
    hidden_dim = 2048
