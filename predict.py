import config
import argparse

from src.detector import Detector
from distutils.util import strtobool

import matplotlib.pyplot as plt
plt.switch_backend("tkAgg")

bool_fn = lambda x: bool(strtobool(str(x)))  # callable parser type to convert string argument to boolean
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default=None, help='Restore specified learning model.')
parser.add_argument("--weight_folder", type=str, default=None, help='Restore specified weights from model restored.')
parser.add_argument("--nms_iou_thresh", default=0.4,
                    help='A threshold value above which two prediction boxes are considered duplicates.')
parser.add_argument("--min_score", default=0.5, help='Minimum class score for predictions.')
parser.add_argument("--rpn_obj_thresh", default=0.6,
                    help='A threshold value above which a prediction is considered valid.')
parser.add_argument("--mask_threshold", default=0.5, help='The threshold value for masking.')
parser.add_argument("--soft_nms_sigma", default=0., help='Soft-NMS parameter. Deactivated if set to 0.')
parser.add_argument("--nms_top_n", default='25', help='Maximum number of objects to detect in from input.')
parser.add_argument("--show_class_names", type=bool_fn, default='True',
                    help='If True, display the class name of predicted objects.')
parser.add_argument(
    "--hide_below_threshold", type=bool_fn, default='True',
    help='If True, predictions whose class scores are below the specified threshold are not displayed.')
parser.add_argument("--hide_bg_class", type=bool_fn, default='False',
                    help='If True, background predictions are not displayed.')
parser.add_argument("--rpn_detection", type=bool_fn, default='False',
                    help='If True, only display object predictions from the region proposal network (RPN).')
parser.add_argument("--add_mask_contour", type=bool_fn, default='True',
                    help='If True, add contour line to generated mask.')
ARGS, unknown = parser.parse_known_args()


if __name__ == '__main__':
    # Initiate the detector object.
    detector_obj = Detector(
        input_size=config.INPUT_SIZE,
        batch_size=config.BATCH_SIZE,
        training_classes=config.CLASSES_TO_LEARN,
        training_categories=config.CATEGORIES_TO_LEARN,
        data_folder=config.DATA_FOLDER,
        num_classes=config.NUM_CLASSES,
        save_path=config.SAVE_WEIGHTS_PATH,
        restore_lr=False,
        load_backbone=True,
        load_rpn=True,
        load_head=True,
        load_cpkt=True)

    # Load variables from trained model.
    detector_obj.load_model_variables(
        restore_lr=False,
        training_classes=config.CLASSES_TO_LEARN,
        load_backbone=config.LOAD_BACKBONE_WEIGHTS,
        load_rpn=config.LOAD_RPN_WEIGHTS,
        load_head=config.LOAD_HEAD_WEIGHTS,
        load_mask=config.LOAD_MASK_WEIGHTS,
        load_cpkt=config.LOAD_CHECKPOINT,
        save_path=config.SAVE_WEIGHTS_PATH,
        model_to_restore=ARGS.model_folder,
        weight_folder_to_restore=ARGS.weight_folder)

    # Predict from random single image.
    img, _, _, _ = detector_obj.data_generator.generate_random_test_data()
    detector_obj.predict(
        image=img[0],
        nms_top_n=int(ARGS.nms_top_n),
        nms_iou_thresh=float(ARGS.nms_iou_thresh),
        min_score=float(ARGS.min_score),
        show_class_names=ARGS.show_class_names,
        hide_below_threshold=ARGS.hide_below_threshold,
        hide_bg_class=ARGS.hide_bg_class,
        soft_nms_sigma=float(ARGS.soft_nms_sigma),
        rpn_detection=ARGS.rpn_detection,
        rpn_obj_thresh=float(ARGS.rpn_obj_thresh),
        add_mask_contour=ARGS.add_mask_contour,
        mask_threshold=float(ARGS.mask_threshold))
