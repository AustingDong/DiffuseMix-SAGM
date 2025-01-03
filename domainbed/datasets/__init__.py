from typing import List, Dict
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from domainbed.algorithms.algorithms import Algorithm
from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT
from domainbed.structs.args import Args
from functools import partial

class _SplitDataset(Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset: Dataset, keys: List[int]):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms: Dict[str, function] = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}

        for key, transform in self.transforms.items():
            # Handle MultipleEnvironmentImageFolderWithAdaptiveDiffusemix
            if isinstance(x, tuple):
                ret[key] = tuple(transform(x))
            else: # Base case
                ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0) -> tuple[_SplitDataset, _SplitDataset]:
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def get_dataset(
    test_envs: List[int],
    args: Args,
    hparams: dict,
    algorithm_class: Algorithm | None = None
) -> tuple[datasets.MultipleEnvironmentImageFolder, list[tuple[_SplitDataset, np.ndarray]], list[tuple[_SplitDataset, np.ndarray]]]:
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset

    diffusemix = 0
    if args.dataset.split("_")[-1] == "Generated":
        dataset: datasets.MultipleEnvironmentImageFolderWithAdaptiveDiffusemix = vars(datasets)[args.dataset](args.data_dir, test_envs, args)
        diffusemix = getattr(args, "diffusemix", 0)
    else:
        dataset: datasets.MultipleEnvironmentImageFolder = vars(datasets)[args.dataset](args.data_dir, args)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        diffusemix_args = None
        if diffusemix:
            diffusemix_args = env.get_diffusemix_args()

        set_transforms(in_, in_type, hparams, algorithm_class, diffusemix, diffusemix_args)
        set_transforms(out, out_type, hparams, algorithm_class, diffusemix, diffusemix_args)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


def set_transforms(dset: _SplitDataset, data_type: str, hparams: dict, algorithm_class: Algorithm | None = None, diffusemix: int = 0, diffusemix_args: dict = None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        # use lambda to pass arguments to the function
        dset.transforms = {"x": partial(DBT.get_aug, diffusemix=diffusemix, diffusemix_args=diffusemix_args)}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.get_basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": partial(DBT.get_aug, diffusemix=diffusemix, diffusemix_args=diffusemix_args)}
    elif data_type == "test":
        dset.transforms = {"x": DBT.get_basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform
