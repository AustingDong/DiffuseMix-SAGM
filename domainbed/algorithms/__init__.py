from .algorithms import *


def get_algorithm_class(algorithm_name: str) -> Algorithm:
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
