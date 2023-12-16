import argparse


def main(args):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the model with user-specified input")
    parser.add_argument("-m")
    args = parser.parse_args()
    main(args)
