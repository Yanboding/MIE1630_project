import ast

import numpy as np
from collections.abc import Iterable

def numpy_shift(arr, num_places, fill_na=0):
    """Shifts the elements of the array to the right by num_places, filling with 0."""
    if num_places == 0:
        return arr
    elif num_places > 0:
        result = np.empty_like(arr, dtype=float)
        result[:num_places] = fill_na
        result[num_places:] = arr[:-num_places]
    else:
        result = np.empty_like(arr, dtype=float)
        result[num_places:] = fill_na
        result[:num_places] = arr[-num_places:]

    return result

def iter_to_tuple(lst):
    if isinstance(lst, Iterable):
        return tuple(iter_to_tuple(sub) for sub in lst)
    return lst

def convert_tuple_keys_to_str(d):
    """ Recursively convert tuple keys to strings. """
    if isinstance(d, dict):
        return {str(k): convert_tuple_keys_to_str(v) for k, v in d.items()}
    else:
        return d

def convert_str_keys_to_tuple(d):
    """ Recursively convert string keys back to tuples. """
    if isinstance(d, dict):
        return {ast.literal_eval(k): convert_str_keys_to_tuple(v) for k, v in d.items()}
    else:
        return d

def keep_significant_digits(number, significant: int) -> float:
    return float(("{:." + str(significant) + "g}").format(number))
