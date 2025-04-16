"""Torch decoder architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocess import BatchTokenizer, SentenceEmbedding
from config.data_dictionary import Decoder_Enum, Train, HuggingFaceData


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_attention_heads,
        hidden_dim,
        drop_prob,
        max_seq_length,
        vocab_to_index,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        self.vocab_to_index = vocab_to_index
        self.PADDING_TOKEN = PADDING_TOKEN
        self.layers = nn.Sequential(
            *[
                Decoder_Block(d_model, num_attention_heads, hidden_dim, drop_prob)
                for _ in range(self.num_layers)
            ]
        )  # Note: Sequential APPLIES the layers in order unlike modulelist layer
        self.sentence_embedding = SentenceEmbedding(
            self.max_seq_length,
            self.d_model,
            self.vocab_to_index,
            self.drop_prob,
            self.PADDING_TOKEN,
        )

    def forward(self, x, y, cross_mask, self_mask, start_token=True, end_token=False):
        # x: 64, 300, 512
        # y: (64, 300)
        # cross_mask: 64, 1, 300, 300
        # self_mask: 64, 1, 300, 300

        y = self.sentence_embedding(y)  # 64, 300, 512
        # Sequential layer takes only one input, hence to use x, y and masks, we need to iterate
        for layer in self.layers:
            y = layer(x, y, cross_mask, self_mask)  # 64, 300, 512

        return y  # 64, 300, 512


class Decoder_Block(nn.Module):
    def __init__(self, d_model, num_attention_heads, hidden_dim, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.self_attention = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.encoder_decoder_attention = MultiHeadCrossAttention(
            self.d_model, self.num_attention_heads
        )
        self.norm1 = LayerNormalization(num_features=self.d_model)
        self.norm2 = LayerNormalization(num_features=self.d_model)
        self.norm3 = LayerNormalization(num_features=self.d_model)

        self.dropout1 = nn.Dropout(self.drop_prob)
        self.dropout2 = nn.Dropout(self.drop_prob)
        self.dropout3 = nn.Dropout(self.drop_prob)
        self.feed_forward = FeedForward(self.d_model, self.hidden_dim, self.drop_prob)

    def forward(self, x, y, cross_mask, self_mask):
        # x: 64, 300, 512, y: 64, 300, 512, cross_mask: 64, 1, 300, 300, self_mask: 64, 1, 300, 300
        residual_y = y
        y = self.self_attention(y, mask=self_mask)  # 64, 300, 512
        y = self.dropout1(y)  #  64, 300, 512
        y = self.norm1(y + residual_y)  # 64, 300, 512

        residual_y = y  # 64, 300, 512
        y = self.encoder_decoder_attention(x, y, mask=cross_mask)  # 64, 300, 512
        y = self.dropout2(y)  # 64, 300, 512
        y = self.norm2(y + residual_y)  # 64, 300, 512

        residual_y = y  # 64, 300, 512
        y = self.feed_forward(y)  # 64, 300, 512
        y = self.dropout3(y)  # 64, 300, 512
        y = self.norm3(y + residual_y)  # 64, 300, 512
        return y


def scaled_dot_product_attention(q, k, v, mask=None):
    # q,k,v: 64, 8, 300, 64
    # mask: 64, 1, 300, 300
    d_k = q.size()[-1]  # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / d_k**0.5  # 64, 8, 300, 300
    if mask is not None:
        scaled += mask  # 64, 8, 300, 300
    attention = F.softmax(scaled, dim=-1)  # 64, 8, 300, 300
    values = torch.matmul(attention, v)  # 64, 8, 300, 64
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = self.d_model // self.num_heads  # 64
        self.qkv_layer = nn.Linear(
            self.d_model, 3 * self.d_model
        )  # Wq, Wk, Wv together
        self.linear_layer = nn.Linear(
            self.d_model, self.d_model
        )  # For cross interaction between multiple heads

    def forward(self, x, mask):
        # x: 64, 300, 512
        # mask: 64, 1, 300, 300
        batch_size, seq_length, d_model = x.size()
        qkv = self.qkv_layer(x)  # 64, 300, 1536
        qkv = qkv.reshape(
            batch_size, seq_length, self.num_heads, 3 * self.head_dim
        )  # 64, 300, 8, 192
        qkv = qkv.permute(0, 2, 1, 3)  # 64, 8, 300, 192
        q, k, v = qkv.chunk(3, dim=-1)  # q,k,v: 64, 8, 300, 64
        values, attention = scaled_dot_product_attention(
            q, k, v, mask
        )  # values :64, 8, 300, 64, attention: 64, 8, 300, 300
        values = values.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )  # 64, 300, 512
        out = self.linear_layer(values)  # 64, 300, 512
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = self.d_model // self.num_heads
        self.q_layer = nn.Linear(self.d_model, self.d_model)
        self.kv_layer = nn.Linear(self.d_model, 2 * self.d_model)
        self.linear_layer = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, y, mask):
        # x: 64, 300, 512
        # y: 64, 300, 512
        # mask: 64, 1, 300, 300
        batch_size, seq_length, d_model = x.size()
        q = self.q_layer(x)  # 64, 300, 512
        kv = self.kv_layer(y)  # 64, 300, 1024
        q = q.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )  # 64, 300, 8, 64
        kv = kv.reshape(
            batch_size, seq_length, self.num_heads, 2 * self.head_dim
        )  # 64, 300, 8, 128
        q = q.permute(0, 2, 1, 3)  # 64, 8, 300, 64
        kv = kv.permute(0, 2, 1, 3)  # 64, 8, 300, 128
        k, v = kv.chunk(2, dim=-1)  # k,v: 64, 8, 300, 64
        values, attention = scaled_dot_product_attention(
            q, k, v, mask
        )  # values :64, 8, 300, 64, attention:  64, 8, 300, 300
        values = values.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )  # 64, 300, 512
        out = self.linear_layer(values)  # 64, 300, 512
        return out


class LayerNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.num_features))  # (512,)
        self.beta = nn.Parameter(torch.zeros(self.num_features))  # (512,)

    def forward(self, x):
        # x: 64, 300, 512
        mean = x.mean(-1, keepdim=True)  # 64, 300, 1
        var = ((x - mean) ** 2).mean(-1, keepdim=True)  # 64, 300, 1
        std = (var + self.eps).sqrt()  # 64, 300, 1
        y = (x - mean) / std  # 64, 300, 512
        out = self.gamma * y + self.beta  # 64, 300, 512 #Broadcasting is used
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.linear1 = nn.Linear(self.d_model, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        # x - (64, 300, 512)
        x = self.linear1(x)  # (64, 300, 2048)
        x = self.relu(x)  #  (64, 300, 2048)
        x = self.dropout(x)  # (64, 300, 2048)
        x = self.linear2(x)  # (64, 300, 512)
        return x


if __name__ == "__main__":

    # Test

    import pickle
    from config.data_dictionary import ROOT
    from pathlib import Path
    from config.data_dictionary import (
        START_TOKEN,
        END_TOKEN,
        UNKNOWN_TOKEN,
        PADDING_TOKEN,
    )

    fp = ROOT / Path("result/preprocessor.pkl")
    with open(fp, "rb") as f:
        preprocessor = pickle.load(f)
    print(preprocessor.ml_vocab_to_index)

    # Test
    x = torch.randn(
        Train.batch_size.value,
        HuggingFaceData.max_length.value,
        Decoder_Enum.d_model.value,
    )  # Src sentence embedding with positional encoding

    y_batch_sentences = [
        "പൂച്ച ഉറങ്ങുകയാണ്.",
        "അവൾ പുസ്തകങ്ങൾ വായിക്കാൻ ഇഷ്ടപ്പെടുന്നു.",
        "അവൻ എല്ലാ ഞായറാഴ്ചയും ഫുട്ബോൾ കളിക്കുന്നു.",
        "സൂര്യൻ ദീപ്തമായി പ്രകാശിക്കുന്നു.",
        "ഞങ്ങൾ പാർക്കിലേക്ക് പോകുകയാണ്.",
        "അവർക്ക് ഒരു വലിയ വീട് ഉണ്ട.",
        "എനിക്ക് കാപ്പി കുടിക്കാൻ ഇഷ്ടമാണ്.",
        "പൂക്കൾ പൂത്തു തുടങ്ങുകയാണ്.",
        "അവൾ ദൈനംദിനം തന്റെ ജേർണലിൽ എഴുതുന്നു.",
        "പ്രഭാതത്തിൽ പക്ഷികൾ പാടുന്നു.",
        "എന്റെ നായ ഓടാൻ ഇഷ്ടപ്പെടുന്നു.",
        "അവൻ ഒരു സിനിമ കാണുകയാണ്.",
        "അവൾ രുചികരമായ ഒരു കേക്ക് ഉണ്ടാക്കി.",
        "ഞങ്ങൾ ഒരു പുതിയ നഗരത്തിലേക്ക് യാത്ര ചെയ്തു.",
        "കാറിന് കൂടുതൽ ഇന്ധനം ആവശ്യമുണ്ട്.",
        "അവൻ രാത്രിയിൽ വൈകിയാണ് പഠിക്കുന്നത്.",
        "കുട്ടികൾ പുറത്തു കളിക്കാൻ ഇഷ്ടപ്പെടുന്നു.",
        "ഇന്ന് കനത്ത മഴയാണ്.",
        "കാറ്റ് ശക്തമായി വീശുന്നു.",
        "എന്റെ സുഹൃത്ത് ഉടൻ സന്ദർശിക്കാനാണ്.",
        "അവൾ ഒരു പുതിയ ലാപ്ടോപ്പ് വാങ്ങി.",
        "കുഞ്ഞ് ഉച്ചത്തിൽ കരയുന്നു.",
        "ഞങ്ങൾ ഇന്നലെ ഒരു ഇന്ദ്രധനുസ്സ് കണ്ടു.",
        "അവൻ നീന്തൽ പഠിക്കുകയാണ്.",
        "ട്രെയിൻ മധ്യാഹ്നത്തിൽ എത്തും.",
        "അവൾ ക്ലാസിക്കൽ സംഗീതം കേൾക്കുന്നു.",
        "ഞങ്ങൾ കഴിഞ്ഞ ആഴ്ച മ്യൂസിയം സന്ദർശിച്ചു.",
        "അവൻ എല്ലായ്പ്പോഴും നേരത്തേ എഴുന്നേൽക്കുന്നു.",
        "എന്റെ സഹോദരി ഒരു മഹാനായ കലാകാരിയാണ്.",
        "ആ പുസ്തകം വളരെ ആകർഷകമായിരുന്നു.",
        "ഞാൻ ഒരു നഷ്ടപ്പെട്ട നായ്ക്കുട്ടിയെ കണ്ടെത്തി.",
        "അവർ ഇന്നലെ ഒരു പാർട്ടി നടത്തുകയാണ്.",
        "അവൾ വേഗത്തിൽ ഹോംവർക്ക് തീർത്ത്.",
        "ഞങ്ങൾ ഒരു ഹാസ്യ സിനിമ കണ്ടു.",
        "അവൻ പർവ്വതങ്ങളിൽ ഹൈക്കിംഗ് ചെയ്യാൻ ഇഷ്ടപ്പെടുന്നു.",
        "സ്റ്റോർ രാത്രി 9 മണിക്ക് അടയ്ക്കും.",
        "അവൾ പുതിയ പച്ചക്കറികൾ വാങ്ങി.",
        "ഞങ്ങൾ ജപ്പാനിലേക്ക് ഒരു യാത്ര ആസൂത്രണം ചെയ്യുകയാണ്.",
        "അവൻ പിയാനോ പ്രാക്ടീസ് ചെയ്യുന്നു.",
        "എനിക്ക് ചോക്ലേറ്റ് കേക്ക് കഴിക്കാൻ ഇഷ്ടമാണ്.",
        "അധ്യാപകൻ പാഠം നന്നായി വിശദീകരിച്ചു.",
        "അവർ ക്രിസ്മസിനായി വീട് അലങ്കരിച്ചു.",
        "അവൾ പസിൽ എളുപ്പത്തിൽ പരിഹരിച്ചു.",
        "ഞങ്ങൾ പാർക്കിൽ ഒരു പിക്നിക് നടത്തി.",
        "അവൻ എല്ലാ രാവിലെ അഞ്ചു മൈൽ ഓടുന്നു.",
        "ശീതകാലത്ത് തടാകം മഞ്ഞിലൂടെയാണ്.",
        "അവൾ മനോഹരമായ ഒരു സൂര്യാസ്തമയം ചിത്രീകരിച്ചു.",
        "ഞങ്ങൾ മുഴുവൻ രാത്രി ബോർഡ് ഗെയിം കളിച്ചു.",
        "വിമാനം മൃദുവായി ഇറങ്ങി.",
        "അവൻ തകർന്ന കസേര ശരിയാക്കി.",
        "അവൾ ഒരു തെരുവ് പൂച്ച ദത്തെടുത്ത്.",
        "അവർ ഒരുമിച്ച് കുക്കീസ് പൊള്ളിക്കുന്നു.",
        "ഞാൻ ഒരു ചെറുകഥ എഴുതുകയാണ്.",
        "പർവ്വത ദൃശ്യങ്ങൾ മനോഹരമാണ്.",
        "അവൾ മനോഹരമായൊരു ഗാനം ആലപിച്ചു.",
        "ഞങ്ങൾ ഒരു ബോട്ട് യാത്രയ്ക്ക് പോയി.",
        "അവൻ ഒരു പുതിയ വെബ്‌സൈറ്റ് ഡിസൈൻ ചെയ്യുന്നു.",
        "നായ അജ്ഞാതനെ കുരച്ചു.",
        "ഞാൻ ഒരു പുതിയ ഭാഷ പഠിക്കുകയാണ്.",
        "അവൾ അതിസുന്ദരമായ ഫോട്ടോകൾ എടുത്തു.",
        "അവർ ഒരു കശുമാവ് വീട്ടിൽ നിർമ്മിച്ചു.",
        "ഞാൻ ശാസ്ത്ര പ്രദർശനം ആസ്വദിച്ചു.",
        "കാപ്പി ഷോപ്പ് തിരക്കേറിയതായിരുന്നു.",
        "അവൻ ഒരു വയോധികനെ റോഡ് കടക്കാൻ സഹായിച്ചു.",
    ]

    tgt_batch_tokenizer = BatchTokenizer(
        HuggingFaceData.max_length.value,
        preprocessor.ml_vocab_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
        UNKNOWN_TOKEN,
    )

    y = tgt_batch_tokenizer(
        y_batch_sentences, start_token=True, end_token=False
    )  # (64, 300)

    cross_mask = torch.ones(
        Train.batch_size.value,
        1,
        HuggingFaceData.max_length.value,
        HuggingFaceData.max_length.value,
    )
    self_mask = torch.full(
        [
            Train.batch_size.value,
            1,
            HuggingFaceData.max_length.value,
            HuggingFaceData.max_length.value,
        ],
        1e-20,
    )
    self_mask = torch.triu(self_mask, diagonal=1)

    decoder = Decoder(
        num_layers=Decoder_Enum.num_layers.value,
        d_model=Decoder_Enum.d_model.value,
        num_attention_heads=Decoder_Enum.num_attention_heads.value,
        hidden_dim=Decoder_Enum.hidden_dim.value,
        drop_prob=Decoder_Enum.drop_prob.value,
        max_seq_length=HuggingFaceData.max_length.value,
        vocab_to_index=preprocessor.ml_vocab_to_index,
        PADDING_TOKEN=PADDING_TOKEN,
    )
    out = decoder(x, y, cross_mask, self_mask, start_token=True, end_token=False)
    print(out.shape)
