"""Directory creation helpers."""

import datetime
import os


def make_dir(dir: str) -> None:
    """Recursively create dir, building each path component individually."""
    dir_lst = dir.split('/')
    for idx in range(1, len(dir_lst) + 1):
        if not os.path.exists(os.path.join(*dir_lst[:idx])):
            os.mkdir(os.path.join(*dir_lst[:idx]))


def make_model_directories(save_dir: str, test_dir=None, test_dir_ext=None) -> str:
    """Create output directories for a run.

    Training (test_dir=None): creates save_dir/ckpts. Testing: creates and returns save_dir/<test_dir>_<ext>.
    """
    if test_dir:
        if test_dir_ext is None:
            test_dir_ext = datetime.datetime.now().strftime("%y%m%d_%H%M")
        save_dir_ = os.path.join(save_dir, f"{test_dir}_{test_dir_ext}")
        make_dir(save_dir_)
        return save_dir_

    else:
        if not os.path.exists(save_dir):
            make_dir(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'ckpts')):
            os.mkdir(os.path.join(save_dir, 'ckpts'))
