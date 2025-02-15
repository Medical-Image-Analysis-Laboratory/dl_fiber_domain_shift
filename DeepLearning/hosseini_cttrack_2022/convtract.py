from test import Test
from train import Train
import argparse
import os

# set the training device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.utils import set_global_seed

# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)


if __name__ == "__main__":

    set_global_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-action", choices=["train", "test"], help="Action to take", default="train"
    )
    parser.add_argument(
        "-data",
        help="path to a directory containing dwi and bvals/bvecs",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-labels",
        help="path to a directory containing ground truth SH.",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-bm", help="path to brain mask", action="store", default=os.getcwd()
    )
    parser.add_argument(
        "-wm", help="path to white matter mask", action="store", default=os.getcwd()
    )
    parser.add_argument(
        "-trained_model_dir",
        metavar="model_dir",
        help="trained model (.ckpt)",
        default=None,
    )
    parser.add_argument(
        "-save_dir",
        help="directory to save trained model or generated tractogram",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-algorithm",
        action="store",
        default="deterministic",
        help="Tractography algorithm (deterministic or probabilistic)",
    )
    parser.add_argument(
        "-num_tracts",
        action="store",
        default=120000,
        help="number of streamlines (default 80000)",
        type=int,
    )
    parser.add_argument(
        "-min_length",
        action="store",
        default=10,
        help="min length of streamlines (default 10 mm)",
        type=int,
    )
    parser.add_argument(
        "-max_length",
        action="store",
        default=250,
        help="max length of streamlines (default 250 mm)",
        type=int,
    )
    parser.add_argument(
        "-train_batch_size",
        action="store",
        default=400,
        help="train atch size",
        type=int,
    )
    parser.add_argument(
        "-track_batch_size",
        action="store",
        default=1000,
        help="track batch size",
        type=int,
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=0.0002, help="learning rate", type=int
    )
    parser.add_argument("--epochs", default=10, help="number of epochs", type=int)
    parser.add_argument(
        "-dropout_prob", default=0.1, help="dropout probability", type=int
    )
    parser.add_argument(
        "-split_ratio", default=0.8, help="train test split ratio", type=int
    )
    parser.add_argument(
        "-grad_directions", default=15, help="number of gradient directions", type=int
    )
    parser.add_argument(
        "-sh_order", default=4, help="spherical harmonics order", type=int
    )
    parser.add_argument("-gt_kind", default="MSMT", help="ground truth kind", type=str)
    parser.add_argument("-age_group", default="early", help="age group", type=str)

    args = parser.parse_args()
    print(args)

    if args.action == "train":
        trainer = Train(args)
        trainer.train()

    else:
        tracker = Test(args)
        tractogram = tracker.track()
