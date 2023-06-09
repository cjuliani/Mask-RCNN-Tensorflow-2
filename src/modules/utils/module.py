import sys
import copy
import importlib


def replace_specified_char(input_string, replaced_by='_', to_replace=None):
    """Replaces special characters.

    Args:
        input_string (str): the string to be analysed.
        replaced_by (str): replace specific characters by the
            provided string.
        to_replace (list or str or None): if provided, replace any
            provided characters. Otherwise, use below-defined standard
            characters.
    """
    if to_replace is None:
        to_replace = ['-', ' ', '@', '!', '"', '{', '}', '<', '>', '?', '.']
        to_replace += ['+', '\'', '=', '(', ')', '[', ']', '#', '\"', '%', '&']
        to_replace += ['\\', ':', ';', ',', '*', '<', '>', '|', '~', '^', '¨']

    if isinstance(to_replace, str):
        # Make sure character to replace is within a list.
        to_replace = [to_replace]

    name_formatted = copy.deepcopy(input_string)
    for char in to_replace:
        name_formatted = name_formatted.replace(char, replaced_by)

    return name_formatted


def get_module(path, model_name='model'):
    """Returns a learning model specified from path."""
    path = replace_specified_char(path, replaced_by="/", to_replace=["\\"])

    # Add module path to system (temporarily) and import module
    sys.path.append(path)   # directory containing modules
    try:
        spec = importlib.util.find_spec(model_name, path)
        if spec is None:
            print("Import error 0: " + f"module from '{path}' not found.")
            print("Error generated by the function 'get_module'.")
            sys.exit(0)
        module = spec.loader.load_module()
    except (ValueError, ImportError) as msg:
        print(f"Import error 3 when loading '{path}': " + str(msg))
        sys.exit(0)

    return module
