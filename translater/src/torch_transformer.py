import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocess import BatchTokenizer
from src.torch_encoder import Encoder
from src.torch_decoder import Decoder
from config.data_dictionary import START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,  # hidden layer dimension in both Encoder and Decoder
        num_layers,  # No. of encoder layers = No. of Decoder layers
        num_attention_heads,  # No. self attention heads in Encoder and Decoder and no. of cross attention heads bw Encoder and Decoder
        hidden_dim,  # FFW hidden layer dimension in both encoder and decoder
        drop_prob,  # Dropout probability in both Encoder and Decoder (Same across any droput layer)
        max_seq_length,  # Maximum sequence length, same in both Src and Target
        src_vocab_to_index,  # Mapping of src language vocab to index
        tgt_vocab_to_index,  # Mapping of tgt language  vocab to index
        PADDING_TOKEN,  # Padded to max sequence length
    ):
        super().__init__()
        self.target_vocab_size = len(tgt_vocab_to_index)

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_attention_heads,
            hidden_dim,
            drop_prob,
            max_seq_length,
            src_vocab_to_index,
            PADDING_TOKEN,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_attention_heads,
            hidden_dim,
            drop_prob,
            max_seq_length,
            tgt_vocab_to_index,
            PADDING_TOKEN,
        )
        self.linear = nn.Linear(d_model, self.target_vocab_size)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.to(self.device)

    def forward(
        self,
        src,  # (batch, max seq len)
        tgt,  # (batch, max seq len)
        encoder_self_attention_mask,  # (batch, 1, max seq len, max seq len) - Encoder Padding mask
        decoder_encoder_cross_attention_mask,  # (batch, 1, max seq len, max seq len) Decoder & Encoder Padding Mask
        decoder_self_attention_mask,  # (batch, 1, max seq len, max seq len) - Decoder Padding mask + No look ahead mask
        enc_start_token=False,
        enc_end_token=False,
        dec_start_token=False,
        dec_end_token=True,
    ):
        encoder_output = self.encoder(
            src,
            encoder_self_attention_mask,
            start_token=enc_start_token,
            end_token=enc_end_token,
        )  # (batch, max seq len, d_model)
        decoder_output = self.decoder(
            encoder_output,
            tgt,
            decoder_encoder_cross_attention_mask,
            decoder_self_attention_mask,
            start_token=dec_start_token,
            end_token=dec_end_token,
        )  # (batch, max seq len, d_model)

        output = self.linear(
            decoder_output
        )  # (batch, max seq len, target vocab size) #logits
        return output


if __name__ == "__main__":

    # Test

    import pickle
    from config.data_dictionary import ROOT
    from pathlib import Path
    from config.data_dictionary import (
        Encoder_Enum,
        Decoder_Enum,
        Train,
        HuggingFaceData,
    )

    fp = ROOT / Path("result/preprocessor.pkl")
    with open(fp, "rb") as f:
        preprocessor = pickle.load(f)
    print(preprocessor.ml_vocab_to_index)

    _x = [
        "The cat is sleeping.",
        "She loves reading books.",
        "He plays soccer every Sunday.",
        "The sun is shining brightly.",
        "We are going to the park.",
        "They have a big house.",
        "I enjoy drinking coffee.",
        "The flowers are blooming.",
        "She writes in her journal daily.",
        "Birds are singing in the morning.",
        "My dog loves to run.",
        "He is watching a movie.",
        "She made a delicious cake.",
        "We traveled to a new city.",
        "The car needs more fuel.",
        "He studies late at night.",
        "Children love playing outside.",
        "It is raining heavily today.",
        "The wind is blowing hard.",
        "My friend is visiting soon.",
        "She bought a new laptop.",
        "The baby is crying loudly.",
        "We saw a rainbow yesterday.",
        "He is learning to swim.",
        "The train arrives at noon.",
        "She listens to classical music.",
        "We visited the museum last week.",
        "He always wakes up early.",
        "My sister is a great artist.",
        "The book was very interesting.",
        "I found a lost puppy.",
        "They are having a party tonight.",
        "She finished her homework quickly.",
        "We watched a funny movie.",
        "He enjoys hiking in the mountains.",
        "The store closes at 9 PM.",
        "She bought fresh vegetables.",
        "We are planning a trip to Japan.",
        "He is practicing the piano.",
        "I love eating chocolate cake.",
        "The teacher explained the lesson well.",
        "They decorated the house for Christmas.",
        "She solved the puzzle easily.",
        "We had a picnic at the park.",
        "He runs five miles every morning.",
        "The lake is frozen in winter.",
        "She painted a beautiful sunset.",
        "We played board games all night.",
        "The airplane landed smoothly.",
        "He fixed the broken chair.",
        "She adopted a stray kitten.",
        "They are baking cookies together.",
        "I am writing a short story.",
        "The mountain view is breathtaking.",
        "She sang a lovely song.",
        "We went on a boat ride.",
        "He is designing a new website.",
        "The dog barked at the stranger.",
        "I am learning a new language.",
        "She took amazing photographs.",
        "They built a wooden treehouse.",
        "I enjoyed the science exhibition.",
        "The coffee shop was crowded.",
        "He helped an old man cross the street.",
    ]

    src_batch_tokenizer = BatchTokenizer(
        HuggingFaceData.max_length.value,
        preprocessor.eng_vocab_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
        UNKNOWN_TOKEN,
    )

    x = src_batch_tokenizer(_x, start_token=False, end_token=False)  # (64, 300)

    _y = [
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

    y = tgt_batch_tokenizer(_y, start_token=True, end_token=False)  # (64, 300)

    encoder_self_mask = torch.ones(
        Train.batch_size.value,
        1,
        HuggingFaceData.max_length.value,
        HuggingFaceData.max_length.value,
    )

    decoder_cross_mask = torch.ones(
        Train.batch_size.value,
        1,
        HuggingFaceData.max_length.value,
        HuggingFaceData.max_length.value,
    )
    decoder_self_mask = torch.full(
        [
            Train.batch_size.value,
            1,
            HuggingFaceData.max_length.value,
            HuggingFaceData.max_length.value,
        ],
        1e-20,
    )
    decoder_self_mask = torch.triu(decoder_self_mask, diagonal=1)

    transformer = Transformer(
        num_layers=Decoder_Enum.num_layers.value,
        d_model=Decoder_Enum.d_model.value,
        num_attention_heads=Decoder_Enum.num_attention_heads.value,
        hidden_dim=Decoder_Enum.hidden_dim.value,
        drop_prob=Decoder_Enum.drop_prob.value,
        max_seq_length=HuggingFaceData.max_length.value,
        src_vocab_to_index=preprocessor.eng_vocab_to_index,  # Mapping of src language vocab to index
        tgt_vocab_to_index=preprocessor.ml_vocab_to_index,  # Mapping of tgt language  vocab to index
        PADDING_TOKEN=PADDING_TOKEN,
    )

    print(
        transformer(
            x, y, encoder_self_mask, decoder_cross_mask, decoder_self_mask
        ).shape
    )
