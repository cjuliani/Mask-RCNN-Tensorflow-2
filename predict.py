import config
from src.detector import Detector

import matplotlib.pyplot as plt
plt.switch_backend("tkAgg")


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
        model_folder_to_restore="model_001",
        model_to_restore="weights_epoch-0_step-0_20230428-1908")

    # Predict from single image.
    img, _, _, _ = detector_obj.data_generator.generate_random_test_data()
    detector_obj.predict(
        image=img[0],
        nms_top_n=25,
        nms_iou_thresh=0.2,
        min_score=0.4,
        show_class_names=True,
        hide_below_threshold=True,
        hide_bg_class=False,
        soft_nms_sigma=0.,
        rpn_detection=False,
        rpn_obj_thresh=0.6,
        add_mask_contour=True,
        mask_threshold=0.5
        )
