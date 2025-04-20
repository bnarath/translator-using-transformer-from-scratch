from typing import Literal, List
from config.data_dictionary import START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN


class CreateVocabulary:
    def __init__(
        self,
        language: Literal["ml", "eng"],
        type: Literal["letter", "word"],
    ):
        if language not in ["ml", "eng"]:
            raise ValueError(f"Invalid language: {language}. Must be 'ml' or 'eng'.")
        if type not in ["letter", "word"]:
            raise ValueError(f"Invalid type: {type}. Must be 'letter' or 'word'.")

        self.language = language
        self.type = type
        self.vocabulary = [START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN]

    def get_vocab(self):

        if self.language == "ml" and self.type == "letter":

            basic = [
                " ",
                "!",
                '"',
                "#",
                "$",
                "%",
                "&",
                "'",
                "(",
                ")",
                "*",
                "+",
                ",",
                "-",
                ".",
                "/",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ":",
                "<",
                "=",
                ">",
                "?",
                "@",
            ]

            # Vowels (Independent)
            vowels = [
                "അ",
                "ആ",
                "ഇ",
                "ഈ",
                "ഉ",
                "ഊ",
                "ഋ",
                "ൠ",
                "ഌ",
                "എ",
                "ഏ",
                "ഐ",
                "ഒ",
                "ഓ",
                "ഔ",
            ]

            # Consonants
            consonants = [
                "ക",
                "ഖ",
                "ഗ",
                "ഘ",
                "ങ",
                "ച",
                "ഛ",
                "ജ",
                "ഝ",
                "ഞ",
                "ട",
                "ഠ",
                "ഡ",
                "ഢ",
                "ണ",
                "ത",
                "ഥ",
                "ദ",
                "ധ",
                "ന",
                "പ",
                "ഫ",
                "ബ",
                "ഭ",
                "മ",
                "യ",
                "റ",
                "ര",
                "ഴ",
                "ല",
                "ള",
                "വ",
                "ശ",
                "ഷ",
                "സ",
                "ഹ",
            ]

            # Chillaksharams (Ending Consonants)
            ending_consonants = ["ൺ", "ൻ", "ർ", "ൽ", "ൾ", "ൿ"]

            # Anusvara, Visarga, Chandrabindu
            special = [
                "ം",
                "ഃ",
                "ँ",
            ]

            # Vowel Signs (Diacritics)
            ext = ["ാ", "ി", "ീ", "ു", "ൂ", "ൃ", "െ", "േ", "ൈ", "ൊ", "ോ", "ൗ", "്"]

            # Malayalam Numerals
            numerals = ["൦", "൧", "൨", "൩", "൪", "൫", "൬", "൭", "൮", "൯"]

            some_unicodes_not_visible_but_present = ["\u200b", "\u200c", "\u200d"]

            # Add all symbols
            self.add_to_vocabulary(
                basic,
                vowels,
                consonants,
                ending_consonants,
                special,
                ext,
                numerals,
                some_unicodes_not_visible_but_present,
            )
            return self.vocabulary

        elif self.language == "eng" and self.type == "letter":
            vocab = [
                " ",
                "!",
                '"',
                "#",
                "$",
                "%",
                "&",
                "'",
                "(",
                ")",
                "*",
                "+",
                ",",
                "-",
                ".",
                "/",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ":",
                "<",
                "=",
                ">",
                "?",
                "@",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "[",
                "\\",
                "]",
                "^",
                "_",
                "`",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "{",
                "|",
                "}",
                "~",
            ]
            self.add_to_vocabulary(vocab)
            return self.vocabulary

    def add_to_vocabulary(self, *args: List[str]):
        for _list in args:
            for char in _list:
                if char not in self.vocabulary:
                    self.vocabulary.append(char)

    def get_tokens(self):
        self.index_to_vocab = {i: v for i, v in enumerate(self.vocabulary)}
        self.vocab_to_index = {v: i for i, v in enumerate(self.vocabulary)}
        return self.vocab_to_index, self.index_to_vocab
