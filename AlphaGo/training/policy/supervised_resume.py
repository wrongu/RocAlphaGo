import os
import json
from AlphaGo.training.policy.supervised import run_training


def resume_training(cmd_line_args=None):
    """Resume training. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser(description='Restart supervised training on a policy network.')
    # required args
    parser.add_argument("directory", help="directory where metadata and weights are saved.")
    # frequently used args
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the directory) to load to resume training. Default: use newest file", default=None)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    # show help or parse arguments
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # get settings from json
    meta_file = os.path.join(args.directory, "metadata.json")

    # empty metadata object
    metadata = {
            "epoch_logs": [],
            "current_batch": 0,
            "current_epoch": 0,
            "best_epoch": 0
        }

    # load data from json file
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            metadata = json.load(f)
    else:
        raise ValueError("Metadata file not found!")

    # determine what weight file to use
    if args.weights is None:
        # newest epoch weight file from json
        weight_file = "weights.{epoch:05d}.hdf5".format(epoch=metadata["current_epoch"])
    else:
        # user weight argument
        weight_file = args.weights

    if args.verbose:
        print("Resume training")
        print("Weight file: " + weight_file)

        # include verbose
        training_arguments = [metadata["model_file"], metadata["training_data"], args.directory,
                              '--weights', weight_file, "-v"]
    else:
        # exclude verbose
        training_arguments = [metadata["model_file"], metadata["training_data"], args.directory,
                              '--weights', weight_file]

    # start training
    run_training(training_arguments)


if __name__ == '__main__':
    resume_training()
