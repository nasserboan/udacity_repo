import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # Your code here
    # For example:
    logger.info("This is a message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.info(f"This is {args.artifact_name}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_name",
        help='Artifact name for W&B',
        type=str,
        required=True
    )

    args = parser.parse_args()

    go(args)