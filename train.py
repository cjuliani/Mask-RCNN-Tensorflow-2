import os
import config
import warnings
import argparse
import tensorflow as tf
import subprocess as sp

from distutils.util import strtobool
from src.detector import Detector


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
warnings.filterwarnings('ignore')  # disable tensor slicing warnings when calculating gradients

bool_fn = lambda x: bool(strtobool(str(x)))  # callable parser type to convert string argument to boolean
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.4, help='Memory fraction of a GPU used for training.')
parser.add_argument("--gpu_allow_growth", type=bool_fn, default='False', help='Memory growth allowed to GPU.')
parser.add_argument("--gpu_device", default='0', help='Define which GPU device to work on.')
parser.add_argument("--soft_placement", type=bool_fn, default='True',
                    help='Automatically choose an existing device to run tensor operations.')
parser.add_argument("--restore_model", type=str, default=None, help='Restore specified learning model.')
parser.add_argument("--restore_lr", type=bool_fn, default='False', help='Load learning rate of restored model.')
parser.add_argument("--run_eagerly", type=bool_fn, default='False', help='Run in graph mode if False.')
parser.add_argument("--reset_summary", type=bool_fn, default='True', help='Replace existing summary files.')
ARGS, unknown = parser.parse_known_args()


def get_gpu_memory():
    """Returns memory info per available GPU."""
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    return [int(x.split()[0]) for i, x in enumerate(memory_free_info)]


if __name__ == '__main__':
    # Restrict GPU usage in tensorflow.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs.
            tf.config.experimental.set_visible_devices(gpus[int(ARGS.gpu_device)], 'GPU')
            for device in gpus:
                tf.config.experimental.set_memory_growth(device, ARGS.gpu_allow_growth)

            # Restrict TensorFlow to only allocate xGB of memory on the first GPU
            gpu_total_memory = get_gpu_memory()[0]
            memory_to_allocate = gpu_total_memory * float(ARGS.gpu_memory)
            tf.config.set_logical_device_configuration(
                gpus[int(ARGS.gpu_device)],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_to_allocate)])

            # Let TensorFlow automatically choose an existing and
            # supported device to run the operations (instead of
            # specifying one).
            tf.config.set_soft_device_placement(ARGS.soft_placement)
            tf.print('GPU device used:', gpus[int(ARGS.gpu_device)], 'with memory fraction:', memory_to_allocate)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            raise e

    # Define detector.
    detector_obj = Detector(
        input_size=config.INPUT_SIZE,
        batch_size=config.BATCH_SIZE,
        training_categories=config.CATEGORIES_TO_LEARN,
        data_folder=config.DATA_FOLDER,
        training_classes=config.CLASSES_TO_LEARN,
        num_classes=config.NUM_CLASSES,
        model_folder_to_restore=config.MODEL_FOLDER_TO_RESTORE,
        model_to_restore=ARGS.restore_model,
        save_path=config.SAVE_WEIGHTS_PATH,
        run_eagerly=ARGS.run_eagerly,
        restore_lr=ARGS.restore_lr,
        load_backbone=config.LOAD_BACKBONE_WEIGHTS,
        load_rpn=config.LOAD_RPN_WEIGHTS,
        load_head=config.LOAD_HEAD_WEIGHTS,
        load_cpkt=config.LOAD_CHECKPOINT)

    # Train model.
    detector_obj.train_model(
        reset_summary=ARGS.reset_summary)
