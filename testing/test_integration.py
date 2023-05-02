import os
import shutil
import unittest
import mock
import tensorflow as tf

import sys
sys.path.append("..")  # add top folder

import test_config
from src.modules.solver import Solver
from src.detector import Detector


# Define configuration parameters to mock. i.e. target
# module with config, config to mock in target, and
# mocking config.
config_param_1 = (
    "src.modules.solver",
    "config",
    test_config)

config_param_2 = (
    "src.modules.model",
    "config",
    test_config)


def mock_configuration(host_module_file, module_to_mock, mocking_module):
    """Mock configuration parameters of a specified target
    module."""
    def decorator(function):
        def wrapper(*args, **kwargs):
            with mock.patch(f"{host_module_file}.{module_to_mock}") as mock_cfg:
                cfg_attr = [item for item in dir(mocking_module) if "__" not in item]
                name_ = mocking_module.__name__.split('.')[-1]  # get name of module
                for attr in cfg_attr:
                    if '@' in str(attr):
                        continue
                    exec(f"type(mock_cfg).{attr} = {name_}.{attr}")
                return function(*args, **kwargs)
        return wrapper
    return decorator


class TestCase(unittest.TestCase):
    """Module testing the detector core functionalities.

    Attributes:
        detector_obj: the detector object.
        nms_top_n (int) the number of boxes to predict per input.
    """

    detector_obj = None

    @classmethod
    @mock_configuration(*config_param_1)
    @mock_configuration(*config_param_2)
    def setUpClass(cls):
        # Define learning model.
        cls.detector_obj = Detector(
            input_size=(512, 512),
            batch_size=test_config.BATCH_SIZE,
            training_categories=test_config.CATEGORIES_TO_LEARN,
            data_folder=os.path.join(
                test_config.PROJECT_PATH, test_config.DATA_FOLDER
            ),
            training_classes=test_config.CLASSES_TO_LEARN,
            num_classes=test_config.NUM_CLASSES,
            model_folder_to_restore=test_config.MODEL_FOLDER_TO_RESTORE,
            model_to_restore=None,
            save_path=test_config.UNIT_TESTS_WEIGHTS_PATH,
            run_eagerly=True,
            load_cpkt=True,
            load_head=True,
            load_rpn=True,
            load_mask=True,
            load_backbone=True)

        # Define number of boxes to predict per input.
        cls.nms_top_n = 20

        # Build variables from 1 call to access the parameters
        # of variables.
        cls.detector_obj.solver.learning_model.rpn.build_variables_from_call(
            last_dimension=cls.detector_obj.solver.learning_model.backbone_model.last_conv_channel_dim)

        cls.detector_obj.solver.learning_model.bbox_head.build_variables_from_call(
            last_dim_of_input=cls.detector_obj.solver.learning_model.backbone_model.last_conv_channel_dim,
            pool_size=test_config.HEAD_PRE_POOL_SIZE * 2)

        cls.detector_obj.solver.learning_model.bbox_masking.build_variables_from_call(
            last_dim_of_input=cls.detector_obj.solver.learning_model.backbone_model.last_conv_channel_dim)

        cls.detector_obj.reset_model_variables()

    @mock_configuration(*config_param_1)
    def load_model_variables(self):
        """Checks that initial and loaded model weights are different."""
        # Get initial model weights.
        initial_weights = self.detector_obj.get_weights_of_learning_model()

        # Call method.
        self.detector_obj.load_model_variables(
            restore_lr=False,
            training_classes=test_config.CLASSES_TO_LEARN,
            load_backbone=True,
            load_rpn=True,
            load_head=True,
            load_mask=True,
            load_cpkt=True,
            save_path=test_config.UNIT_TESTS_WEIGHTS_PATH,
            model_folder_to_restore=test_config.MODEL_FOLDER_TO_RESTORE,
            model_to_restore=test_config.MODEL_TO_RESTORE)

        loaded_weights = self.detector_obj.get_weights_of_learning_model()

        # Check that loaded and initialized model
        # weights are different.
        cnt = 0
        conditions = []
        for init_w, loaded_w in zip(initial_weights, loaded_weights):
            conditions.append(tf.math.reduce_any(init_w != loaded_w))

        msg = f"Loaded weights must be different (list position: {cnt})"
        self.assertTrue(tf.math.reduce_any(conditions), msg)
        cnt += 1

    @mock_configuration(*config_param_1)
    @mock.patch.object(
        Solver, "write_loss_summaries",
        side_effect=Solver.write_loss_summaries, autospec=True)
    @mock.patch.object(
        Solver, "write_metric_summaries",
        side_effect=Solver.write_metric_summaries, autospec=True)
    def test_model_training(self, mock_metric, mock_loss):
        """Checks that the model learns."""
        # Load model parameters.
        self.detector_obj.load_model_variables(
            restore_lr=False,
            training_classes=test_config.CLASSES_TO_LEARN,
            load_backbone=True,
            load_rpn=True,
            load_head=True,
            load_mask=True,
            load_cpkt=True,
            save_path=test_config.UNIT_TESTS_WEIGHTS_PATH,
            model_folder_to_restore=test_config.MODEL_FOLDER_TO_RESTORE,
            model_to_restore=test_config.MODEL_TO_RESTORE)

        # Call method.
        self.detector_obj.data_generator.data_size = 1  # only process 1 step
        self.detector_obj.train_model(
            reset_summary=False)

        # Check that both, training and validation steps were done.
        msg = "Metric writer must be called twice (training and validation)."
        self.assertEqual(mock_metric.call_count, 2, msg)

        # Check that loss and metric values are realistic.
        method_args = mock_loss.call_args.kwargs

        # Check that number of possible 'nan' is not above 3.
        # 'nan' occurs when not positive boxes are found.
        tensor_ = method_args["values"]
        num_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(tensor_), tf.float32))
        self.assertLessEqual(num_nan, 3, "Number of possible 'nan' must be less or equal to 4.")

        # Check that loss values are within a realistic value range.
        values = mock_loss.call_args.kwargs["values"].numpy().tolist()
        for val in values:
            if val == val:
                msg = f"Loss value {val} is not realistic (negative value)."
                self.assertGreaterEqual(val, 0, msg)

        # Check that metric values are between 0 and 1
        values = mock_metric.call_args.kwargs["values"]
        values = [item.numpy() for item in values]
        for val in values:
            if val == val:
                msg = f"Metric value {val} is not realistic."
                self.assertLessEqual(val, 1., msg)
                self.assertGreaterEqual(abs(val), 0., msg)

        _, directories, _ = os.walk(test_config.RESULTS_FOLDER).__next__()
        self.assertEqual(len(directories), 2, "2 directories must be saved ('weights' and 'summary').")

        # Check weights.
        path = os.path.join(test_config.SAVE_WEIGHTS_PATH, "model_001")
        self.assertTrue(os.path.exists(path), "A weight folder must exist for the trained model.")
        _, subdirs, _ = os.walk(path).__next__()
        _, _, files = os.walk(os.path.join(path, subdirs[0])).__next__()
        self.assertGreater(len(files), 0, f"The weight folder must not be empty.")

        # Check summaries.
        path = os.path.join(test_config.SUMMARY_PATH, "model_001")
        self.assertTrue(os.path.exists(path), "A summary folder must exist for the trained model.")
        _, subdirs, _ = os.walk(path).__next__()
        self.assertEqual(len(subdirs), 2, "2 directories must exists for summary.")
        for dir_ in subdirs:
            _, _, files = os.walk(os.path.join(path, dir_)).__next__()
            self.assertGreater(len(files), 0, f"Folder '{dir_}' must not be empty.")

    @mock_configuration(*config_param_1)
    def model_testing(self):
        """Checks that validation of the model is processed
        correctly."""
        # Load model parameters.
        self.detector_obj.load_model_variables(
            restore_lr=False,
            training_classes=test_config.CLASSES_TO_LEARN,
            load_backbone=True,
            load_rpn=True,
            load_head=True,
            load_mask=True,
            load_cpkt=True,
            save_path=test_config.UNIT_TESTS_WEIGHTS_PATH,
            model_folder_to_restore=test_config.MODEL_FOLDER_TO_RESTORE,
            model_to_restore=test_config.MODEL_TO_RESTORE)

        # Mock detector property.
        # Note: if you replace Foo.my_method with a method that
        # calls Foo.my_method, you get infinite recursion loop.
        # To avoid this, you need to store original method before
        # you patch it.
        original = self.detector_obj.solver.learning_model  # an unbound function; it won't be passed in self
        with mock.patch.object(self.detector_obj.solver, "learning_model") as mock_model:
            mock_model.side_effect = original

            # Call method.
            test_ids = self.detector_obj.solver.data_generator.test_ids[:3]  # only process 3 test data
            self.detector_obj.test_model(input_ids=test_ids)

            self.assertEqual(mock_model.call_count, 3, "The model must be called for every test data.")

            # Check that loss and metric values are realistic.
            method_args = mock_model.call_args.kwargs

        # Check that image input has correct shape and is normalized.
        tensor_ = method_args["inputs"]
        self.assertEqual(tensor_.shape, (1, 512, 512, 3), "Image input tensor has wrong shape.")
        msg = "Tensor of input image must be normalized between 0 and 1."
        self.assertGreaterEqual(tf.reduce_max(tensor_).numpy(), 0., msg)
        self.assertLessEqual(tf.reduce_max(tensor_).numpy(), 1., msg)

        # Check that label input is not be empty.
        tensor_ = method_args["gt_boxes"]
        self.assertEqual(tensor_.numpy().ndim, 3, "Number of dimension for input box tensor must be 3.")
        self.assertGreater(tensor_.numpy().shape[-1], 0, "At least 1 ground truth input must be generated.")

    def tearDown(self):
        # Reset model parameters.
        self.detector_obj.reset_model_variables()

    @classmethod
    def tearDownClass(cls):
        # Delete eventual saving directories created during training.
        _, directories, _ = os.walk(test_config.RESULTS_FOLDER).__next__()
        if directories:
            for folder_name in ["save", "summary"]:
                path_to_results = os.path.join(test_config.RESULTS_FOLDER, folder_name)
                if os.path.exists(path_to_results):
                    shutil.rmtree(path_to_results)


if __name__ == "__main__":
    unittest.main()
