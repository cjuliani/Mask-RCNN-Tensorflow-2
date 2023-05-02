import os


def get_folder_or_create(path, name=None):
    """Returns folder path. Also creates folder if it does not exist."""
    if name is None:
        out_path = path
    else:
        out_path = os.path.join(path, name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path
