"Implementation of English to Malayalam translator - letter by letter using Transformer Architecture from Scratch in PyTorch"

import torch
import torch.nn as nn
import fsspec

from src.preprocess import Processor, BatchPadder, BPE
from src.torch_transformer import Transformer
from data.retrieve import Retriever
from utils.utils import get_device

# from utils.print import display_first_n_batch
from config.data_dictionary import ROOT, HuggingFaceData, Train, Decoder_Enum, BPE_Enum
from torch.utils.data import DataLoader, Dataset
from utils.utils import (
    get_checkpoint_path,
    get_log_dir,
    get_bpe_path,
    get_preprocessor_path,
)
from config.data_dictionary import (
    START_TOKEN,
    END_TOKEN,
    PADDING_TOKEN,
    NEG_INFINITY,
)

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from pathlib import Path
import os
import pickle
import time
import logging
from typing import Tuple, List, Dict, Literal

torch.manual_seed(Train.seed.value)


class Translator:
    def __init__(
        self,
        learning_rate=Train.learning_rate.value,
        batch_size=Train.batch_size.value,
        num_epochs=Train.num_epochs.value,
    ):
        self.checkpoint_path = get_checkpoint_path()
        self.log_dir = get_log_dir()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.device = get_device()
        self.device_count = torch.cuda.device_count()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.num_layers = Decoder_Enum.num_layers.value
        self.d_model = Decoder_Enum.d_model.value
        self.num_attention_heads = Decoder_Enum.num_attention_heads.value
        self.hidden_dim = Decoder_Enum.hidden_dim.value
        self.drop_prob = Decoder_Enum.drop_prob.value
        self.max_seq_length = HuggingFaceData.max_length.value
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.NEG_INFINITY = NEG_INFINITY
        self.special_tokens = BPE_Enum.special_tokens.value

        # Data, Vocabulary, training & test dataloader with batches
        self.bpe = self.get_bpe()

        self.batch_padder = BatchPadder(
            self.max_seq_length,
            self.START_TOKEN,
            self.END_TOKEN,
            self.PADDING_TOKEN,
        )
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.get_input()
        )  # Each batch is [[(64, src, 64 tgt)]] where src and tgt are encoded sendences List of integers (variable lengths)

        self.transformer = Transformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            hidden_dim=self.hidden_dim,
            drop_prob=self.drop_prob,
            max_seq_length=self.max_seq_length,
            PADDING_TOKEN=self.PADDING_TOKEN,
        )

        # 🚀 Enable multi-GPU if available
        # DataParallel splits each batch across GPUs automatically.
        # Weights are shared (same weights)
        # Fwd pass and backward separately and gradients are taken as the avg
        # Weights are updated (shared) and repeat the process
        # Eval is not distributed
        if self.device_count > 1:
            logging.info(f"Using {self.device_count} GPUs for training!")
            self.transformer = torch.nn.DataParallel(self.transformer)

        self.transformer.to(device=self.device)

        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.special_tokens[self.PADDING_TOKEN], reduction="none"
        )

        # expect logits as input : (batch, max_sequence_len, tgt vocab size)
        # with reduction='none', output is (batch, max_sequence_len)
        # with reduction='mean', output is a single value
        # expects logits, targets in the shape (batch * max seq len, vocab size) vs (batch * max seq len,)
        # With ignore_index, loss corresponsing is kept as 0

        self.initialize_weights(self.transformer)
        # xavier_uniform_ Helps prevent vanishing or exploding gradients.

        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(), lr=self.learning_rate
        )

    def initialize_weights(self, model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def log_model_graph_to_tensorboard(self):
        # dummy inputs are just dummy, shape should be correct
        dummy_src = torch.randint(0, 100, (1, self.max_seq_length)).to(self.device)
        dummy_tgt = torch.randint(0, 100, (1, self.max_seq_length)).to(self.device)
        encoder_self_attention_mask = torch.ones(
            (1, 1, self.max_seq_length, self.max_seq_length)
        ).to(self.device)
        decoder_cross_attention_mask = torch.ones(
            (1, 1, self.max_seq_length, self.max_seq_length)
        ).to(self.device)
        decoder_self_attention_mask = torch.ones(
            (1, 1, self.max_seq_length, self.max_seq_length)
        ).to(self.device)
        self.writer.add_graph(
            self.transformer,
            (
                dummy_src,
                dummy_tgt,
                encoder_self_attention_mask,
                decoder_cross_attention_mask,
                decoder_self_attention_mask,
            ),
            verbose=True,
        )

    def get_collate_fn(self, type: Literal["train", "val", "test"]):
        """Used inside dataloader to collate batches as equal len with padding"""

        if type == "train":

            def collate_fn(batch):
                src_batch, tgt_batch = zip(*batch)
                src_padded = self.batch_padder(
                    src_batch, start_token=False, end_token=False
                )
                tgt_padded = self.batch_padder(
                    tgt_batch, start_token=False, end_token=True
                )
                return src_padded, tgt_padded

        else:  # ["val", "test"]

            def collate_fn(batch):
                src_batch, tgt_batch = zip(*batch)
                src_padded = self.batch_padder(
                    src_batch, start_token=False, end_token=False
                )
                tgt_padded = self.batch_padder(
                    tgt_batch, start_token=True, end_token=False
                )
                label_padded = self.batch_padder(
                    tgt_batch, start_token=False, end_token=True
                )
                return src_padded, tgt_padded, label_padded

        return collate_fn

    def build(self):
        # logging.info(self.transformer)
        # logging.info(list(self.transformer.parameters()))

        start_time = time.time()

        start_epoch = 0
        best_loss = float("inf")
        checkpoint_path = str(self.checkpoint_path)
        check_point_path_exists = False
        if checkpoint_path.startswith("gs://"):
            # Use fsspec for (GCS)
            fs = fsspec.filesystem("gcs")
            if fs.exists(checkpoint_path):
                with fs.open(checkpoint_path, "rb") as f:
                    checkpoint = torch.load(f, map_location=self.device)
                    check_point_path_exists = True

        else:
            # Local filesystem
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                check_point_path_exists = True
        if check_point_path_exists:
            self.transformer.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            logging.info(f"Resuming from epoch {start_epoch}, best loss: {best_loss}")

        for epoch in range(start_epoch, self.num_epochs):
            logging.info(f"Epoch {epoch}")
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.train_dataloader):
                # batch : [(64, ), (64, )]
                # self.transformer.train()  # Train mode : Dropout, Batch norm on (Batch norm not applicable in our case)
                self.transformer.train()
                src, tgt = batch  # (64, 300), (64, 300)

                src.to(self.device)  # (64, 300)

                tgt.to(self.device)  # (64, 300)

                (
                    encoder_self_attention_mask,
                    decoder_self_attention_mask,
                    decoder_cross_attention_mask,
                ) = self.create_masks(
                    (
                        src,
                        tgt,
                    ),  # [(64, 300), (64, 300)],
                    self.max_seq_length,
                    self.special_tokens[self.PADDING_TOKEN],
                    self.special_tokens[self.PADDING_TOKEN],
                )

                self.optimizer.zero_grad()  # resets gradient
                logits = self.transformer(
                    src,
                    tgt,
                    encoder_self_attention_mask.to(self.device),
                    decoder_cross_attention_mask.to(self.device),
                    decoder_self_attention_mask.to(self.device),
                ).to(self.device)
                labels = self.batch_padder(tgt, start_token=False, end_token=True).to(
                    self.device
                )

                loss = self.loss(
                    logits.view(-1, logits.shape[-1]), labels.view(-1)
                )  # flattens batch size, seq len
                non_padding_indices = (
                    labels.view(-1) != self.special_tokens[self.PADDING_TOKEN]
                ).to(self.device)
                loss = (
                    loss.sum() / non_padding_indices.sum()
                )  # Loss of padding indices will be zero
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 1000 == 0:
                    logging.info(f"Testing @ epoch = {epoch}, batch = {batch_idx}")
                    self.test(epoch)

                    # To log the loss
                    self.writer.add_scalar(
                        "Loss/Train",
                        loss.item(),
                        epoch * len(self.train_dataloader) + batch_idx,
                    )

            #     break
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logging.info(f"Average training loss @epoch {epoch} is {avg_epoch_loss}")
            avg_validation_loss = self.get_validation_loss()  # per epoch
            logging.info(
                f"Average validation loss @epoch {epoch} is {avg_validation_loss}"
            )

            # Log average loss for the epoch
            self.writer.add_scalar("Loss/Train_avg", avg_epoch_loss, epoch)
            self.writer.add_scalar("Loss/Test_avg", avg_validation_loss, epoch)

            # log param weights and grads
            for name, param in self.transformer.named_parameters():
                self.writer.add_histogram(
                    f"Params/{name}", param, epoch
                )  # Log parameter histograms
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Grads/{name}", param.grad, epoch
                    )  # Log gradient histograms

            if avg_validation_loss < best_loss:
                best_loss = avg_validation_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.transformer.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                checkpoint_path = str(self.checkpoint_path)
                if checkpoint_path.startswith("gs://"):  # If path is a GCS URI
                    with fsspec.open(checkpoint_path, "wb") as f:
                        torch.save(checkpoint, f)
                else:  # Local filesystem
                    torch.save(checkpoint, checkpoint_path)
                logging.info(
                    f"Checkpoint saved at epoch {epoch} with best validation loss {best_loss:.4f}"
                )
            # break
        end_time = time.time()
        # Compute total training time
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        logging.info(
            f"Total Training Time to run {self.num_epochs-start_epoch} epochs:  {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )

        # Close the TensorBoard writer
        self.writer.close()

    def get_bpe(self) -> BPE:
        """Retrieve the saved trained BPE object"""
        path = get_bpe_path()
        with open(path, "rb") as f:
            bpe_artifacts = pickle.load(f)

        bpe = BPE(corpus=[])
        bpe.map = bpe_artifacts["map"]
        bpe.reverse_map = bpe_artifacts["reverse_map"]

        # Add special tokens to the reverse_map for decoding
        bpe.reverse_map.update(
            {
                v: bytes(k.encode("utf-8"))
                for k, v in BPE_Enum.special_tokens.value.items()
            }
        )

        bpe.vocab_size = bpe_artifacts["vocab_size"]
        return bpe

    def get_input(self) -> Tuple[DataLoader]:
        """Torch dataset for training/ val/ test"""

        processor_path = get_preprocessor_path()
        if os.path.exists(processor_path):
            with open(processor_path, "rb") as f:
                processor = pickle.load(f)
        else:
            processor = Processor()
            processor.process(self.bpe, to_print=True)
            with open(processor_path, "wb") as f:
                pickle.dump(processor, f)

        train_ds = TranslationDataset(src_trg_pairs=processor.valid_train_token_pairs)
        val_ds = TranslationDataset(src_trg_pairs=processor.valid_val_token_pairs)
        test_ds = TranslationDataset(src_trg_pairs=processor.valid_test_token_pairs)

        train_dataloader = DataLoader(
            train_ds,
            batch_size=Train.batch_size.value,
            shuffle=True,
            drop_last=True,
            collate_fn=self.get_collate_fn(type="train"),
        )

        val_dataloader = DataLoader(
            val_ds,
            batch_size=Train.batch_size.value,
            shuffle=True,
            drop_last=True,
            collate_fn=self.get_collate_fn(type="val"),
        )

        test_dataloader = DataLoader(
            test_ds,
            batch_size=Train.batch_size.value,
            shuffle=True,
            drop_last=True,
            collate_fn=self.get_collate_fn(type="test"),
        )

        return train_dataloader, val_dataloader, test_dataloader

    def detockenize(
        self,
        tokens: List[int],
        id_to_vocab: Dict[int, str],
    ) -> str:
        sentence = []
        for token in tokens:
            sentence.append(id_to_vocab[token])
        return "".join(sentence)

    def create_masks(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],  # [(64, 300), (64, 300)]
        max_seq_len: int,
        src_padding_token_id,
        tgt_padding_token_id,
    ):
        src_batch_tokens, tgt_batch_tokens = (
            batch  # src_batch_tokens, tgt_batch_tokens are tensors with shape [(64, 300), (64, 300)]
        )

        batch_size = src_batch_tokens.size(0)

        # Initialize mask
        encoder_padding_mask = torch.full(
            [batch_size, max_seq_len, max_seq_len], fill_value=False, dtype=torch.bool
        )
        decoder_padding_mask = torch.full(
            [batch_size, max_seq_len, max_seq_len], fill_value=False, dtype=torch.bool
        )
        cross_encoder_padding_mask = torch.full(
            [batch_size, max_seq_len, max_seq_len], fill_value=False, dtype=torch.bool
        )
        decoder_lookahead_mask = torch.full(
            [max_seq_len, max_seq_len], fill_value=True, dtype=torch.bool
        )
        decoder_lookahead_mask = torch.triu(decoder_lookahead_mask, diagonal=1)

        # Create padding mask
        src_sent_len = (src_batch_tokens == src_padding_token_id).int().argmax(dim=-1)
        tgt_sent_len = (tgt_batch_tokens == tgt_padding_token_id).int().argmax(dim=-1)

        for i in range(batch_size):
            encoder_padding_mask[i, :, src_sent_len[i] :] = (
                True  # Influence of all pads on every sentence
            )
            encoder_padding_mask[i, src_sent_len[i] :, :] = (
                True  # Pads towards all other
            )
            decoder_padding_mask[i, :, tgt_sent_len[i] :] = True
            decoder_padding_mask[i, tgt_sent_len[i] :, :] = True
            # Note: In case of cross encoder, Q is from decoder side and K, V are from encoder side. Hence weigh matrix row is decoder side and column is encoder side
            cross_encoder_padding_mask[i, :, src_sent_len[i] :] = True
            cross_encoder_padding_mask[i, tgt_sent_len[i] :, :] = True

        encoder_self_attention_mask = torch.where(
            encoder_padding_mask, self.NEG_INFINITY, 0
        )
        decoder_cross_attention_mask = torch.where(
            cross_encoder_padding_mask, self.NEG_INFINITY, 0
        )
        decoder_self_attention_mask = torch.where(
            decoder_padding_mask | decoder_lookahead_mask, self.NEG_INFINITY, 0
        )  # bool OR operation with broadcasting

        return (
            encoder_self_attention_mask.unsqueeze(dim=1),
            # Expecting dim as (batch size, 1, max seq len, max seq len)
            decoder_self_attention_mask.unsqueeze(dim=1),
            decoder_cross_attention_mask.unsqueeze(dim=1),
        )

    def test(self, epoch):
        # Inference test on a specific english sentence
        self.transformer.eval()  # no dropout

        eg_ml = ""  # placeholder
        eg_ml_tokens = [self.special_tokens[self.START_TOKEN]]
        eg_english = "It was fun coding the transformer from scratch"

        with torch.no_grad():  # Disable gradient computation for efficiency

            eg_english_tokens = self.bpe.encode(eg_english)

            eg_english_tokenized = self.batch_padder(
                [eg_english_tokens], start_token=False, end_token=False
            ).to(
                self.device
            )  # (1, 300)

            for i in range(self.max_seq_length - 1):  # To consider start token

                eg_ml_tokenized = self.batch_padder(
                    [eg_ml_tokens], start_token=True, end_token=False
                ).to(
                    self.device
                )  # (1, 300)
                (
                    eg_encoder_self_attention_mask,
                    eg_decoder_self_attention_mask,
                    eg_decoder_cross_attention_mask,
                ) = self.create_masks(
                    [eg_english_tokenized, eg_ml_tokenized],
                    self.max_seq_length,
                    self.special_tokens[self.PADDING_TOKEN],
                    self.special_tokens[self.PADDING_TOKEN],
                )

                eg_logits = self.transformer(
                    eg_english_tokenized,
                    eg_ml_tokenized,
                    eg_encoder_self_attention_mask.to(self.device),
                    eg_decoder_cross_attention_mask.to(self.device),
                    eg_decoder_self_attention_mask.to(self.device),
                ).to(self.device)

                next_token_logit_distribution = eg_logits[0][i]
                # next word of ith word's logit - shape is (vocab_size, )

                next_token_index = torch.argmax(next_token_logit_distribution).item()
                eg_ml_tokens.append(next_token_index)

                if next_token_index == self.special_tokens[self.END_TOKEN]:
                    break
            eg_ml = self.bpe.decode(eg_ml_tokens[1:])
            logging.info(f"{eg_english} -> {eg_ml}")
            logging.info("_________________________________________________________")
            self.writer.add_text(
                "Test Translation", f"Input: {eg_english}\nOutput: {eg_ml}", epoch
            )

    def get_validation_loss(self):
        self.transformer.eval()  # Set model to evaluation mode
        total_loss = 0.0

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch in self.val_dataloader:
                src, tgt, label = batch

                src.to(self.device)
                # (64, 300)

                tgt.to(self.device)
                # (64, 300)

                label.to(self.device)
                # (64, 300)

                (
                    encoder_self_attention_mask,
                    decoder_self_attention_mask,
                    decoder_cross_attention_mask,
                ) = self.create_masks(
                    (src, tgt),
                    self.max_seq_length,
                    self.special_tokens[self.PADDING_TOKEN],
                    self.special_tokens[self.PADDING_TOKEN],
                )

                logits = self.transformer(
                    src,
                    tgt,
                    encoder_self_attention_mask.to(self.device),
                    decoder_cross_attention_mask.to(self.device),
                    decoder_self_attention_mask.to(self.device),
                ).to(self.device)

                loss = self.loss(logits.view(-1, logits.shape[-1]), label.view(-1))

                non_padding_indices = (
                    label.view(-1) != self.special_tokens[self.PADDING_TOKEN]
                ).to(self.device)

                loss = (
                    loss.sum() / non_padding_indices.sum()
                )  # Average loss per word in that batch

                total_loss += loss.item()

        return total_loss / len(
            self.val_dataloader
        )  # Return average validation loss (avg loss per word)


class TranslationDataset(Dataset):
    def __init__(self, src_trg_pairs: list):
        super().__init__()
        self.src_trg_pairs = src_trg_pairs

    def __len__(self):
        return len(self.src_trg_pairs)

    def __getitem__(self, index):
        return self.src_trg_pairs[index]


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    translator = Translator()
    translator.build()
