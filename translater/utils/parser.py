import argparse


def parse():
    p = argparse.ArgumentParser(description="arg parser of main script")
    p.add_argument(
        "--framework",
        dest="framework",
        type=str,
        required=True,
        choices=["pytorch", "tensorflow"],
        help="Choose the framework: pytorch or tensorflow",
    )
    p.add_argument(
        "--type",
        dest="type",
        type=str,
        required=True,
        choices=["letter_by_letter", "BPE"],
        help="Choose the type: letter_by_letter or BPE",
    )
    return p.parse_args()
