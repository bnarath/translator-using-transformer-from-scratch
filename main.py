from src.preprocess import Preprocessor
from src.torch_translator import Translator
import logging


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    preprocessor = Preprocessor()
    preprocessor.process()

    translator = Translator()
    translator.build()
