import os
import time
import warnings
import config
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import namedtuple
from datetime import datetime
from src.modules.utils.folders import get_folder_or_create
from src.modules.utils.summary import get_grid_and_boxes, image_to_figure, plot_to_image
from src.modules.model import Model

plt.switch_backend("Agg")  # make sure no images from summary are plotted when training in eager mode

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
warnings.filterwarnings('ignore')  # disable tensor slicing warnings when calculating gradients


class Solver(object):
    """Trainer module iterating the training and testing processes.
    This module has a learning model defined.

    Attributes:
        batch_size (int): the batch size.
        data_generator (object): the training data generator.
        learning_rate: the learning rate used in training.
        summary_path (str): the path folder to save summary results.
        summary_train_path (str): the saving path for summaries of
            training losses and metrics.
        summary_validation_path (str): the saving path for summaries
            of validation losses and metrics.
        sum_train_writer (object or None): a summary file writer for
            training.
        sum_validation_writer (object or None): a summary file writer
            for validation.
        model_dir (str): the path to model folder where e.g. model
            weights can be saved.
        learning_prop (namedtuple): properties of learning rate
            considered for the training process.
        ckpt: a tensorflow checkpoint used to save
            the learning rate, class names and class values.
        learning_model (object): the learning model to be trained,
            collected from the learning module.
    """

    learning_model = None

    def __init__(self, batch_size, data_generator, num_classes):
        self.batch_size = batch_size
        self.data_generator = data_generator
        self.learning_rate = tf.Variable(
            config.LEARNING_RATE, trainable=False, name="learning_rate")
        self.summary_path = get_folder_or_create(
            path=config.SUMMARY_PATH,
            name=config.MODEL_NAME)
        self.summary_train_path = os.path.join(self.summary_path, 'train')
        self.summary_validation_path = os.path.join(self.summary_path, 'validation')
        self.sum_train_writer = None
        self.sum_validation_writer = None
        self.model_dir = get_folder_or_create(
            path=config.SAVE_WEIGHTS_PATH,
            name=config.MODEL_NAME)

        # Define learning properties.
        props = ['learning_rate', 'lr_decay', 'decay_steps', 'decay_rate',
                 'decay_staircase', 'minimum_lr', 'momentum', 'nesterov',
                 'lr_cyclic', 'cyclic_step_size', 'cyclic_gamma']
        values = [config.LEARNING_RATE, config.LR_DECAY, config.LR_DECAY_STEPS,
                  config.LR_DECAY_RATE, config.LR_DECAY_STAIRCASE,
                  config.MINIMUM_LEARNING_RATE, config.MOMENTUM, config.NESTEROV,
                  config.LR_CYCLIC, config.CYCLIC_STEP_SIZE, config.CYCLIC_GAMMA]
        Properties = namedtuple('learning_properties', props)
        self.learning_prop = Properties(*values)

        # Build checkpoint object to save learning rate center=(90000, 51400),
        # (useful when learning rate decay activated).
        self.ckpt = tf.train.Checkpoint(
            learning_rate=self.learning_rate,
            class_names=tf.Variable([], dtype=tf.string, trainable=False, shape=tf.TensorShape(None)),
            class_values=tf.Variable([], dtype=tf.int32, trainable=False, shape=tf.TensorShape(None)))

        # Build the model from module.
        self.learning_model = Model(
            num_classes=num_classes,
            input_size=config.INPUT_SIZE,
            learning_prop=self.learning_prop,
            batch_size=self.batch_size,
            metric_interval=tf.constant(config.METRIC_RESET_STEP, dtype=tf.float32),
            adam_optimizer=config.ADAM_OPTIMIZER)

        # Assign class names to model.
        self.learning_model.class_names.assign(self.data_generator.class_names)

    @staticmethod
    def add_image_to_summary(figure, file_writer, step, name, cmap, convert_as_fig,
                             norm=False, fig_size=(512, 512)):
        """Writes image summary to tensorboard.

        Args:
            figure (object): matplotlib figure.
            file_writer: a summary file writer
            step (int): current training step.
            name (str): name of the figure.
            cmap (object): color map.
            convert_as_fig (bool): if True, convert the input
                image as a matplotlib figure.
            norm (bool): if True, normalize the image pixels.
            fig_size (tuple): size of the figure considered
                for displaying.
        """
        # Get figure.
        if convert_as_fig:
            figure = image_to_figure(figure, cmap=cmap)  # returns figure with color gradient
        # Convert figure into image to add in tensorboard.
        image = plot_to_image(figure)  # (h, w, 4)
        image = tf.expand_dims(image, axis=0)
        # Resize image.
        if fig_size is not None:
            if norm:
                image = image / 255
                method = "nearest"
            else:
                method = "bilinear"
            image = tf.image.resize(
                images=image,
                size=fig_size,
                method=method,
                preserve_aspect_ratio=True)
        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image(name, image, max_outputs=1, step=step)

    @staticmethod
    def get_box_overview_figure(gt_bbox, pred_bbox, background):
        """Returns a figure showing boxes.

        Args:
            gt_bbox (tensor): ground truth bounding boxes.
            pred_bbox (tensor): predicted bounding boxes.
            background (tensor): image background into which boxes
                are plotted.
        """
        figures = []
        for prop in pred_bbox:
            # Select boxes based on scores.
            indices = tf.argsort(prop[1], direction='DESCENDING')[:config.SUM_NUM_BOXES_TO_SHOW]
            selected_scores = tf.gather(prop[1], indices)
            selected_bbox = tf.gather(prop[0], indices)
            # Get figure with boxes.
            _fig = get_grid_and_boxes(
                background=background,
                gt_boxes=gt_bbox,
                anchors=selected_bbox,
                scores=tf.cast(selected_scores, tf.float32),
                grid_points=None,
                pos_min_score=config.SUM_BOXES_MIN_SCORE,
                show_fig=False)
            figures.append([_fig, prop[2]])
        return figures

    @staticmethod
    def write_loss_summaries(values, other_scalars, writer, step):
        """Write loss summaries to tensorboard.

        Args:
            values (tensor): the loss vector.
            other_scalars (list): list of scalar tensors to write,
                other than losses.
            writer: a summary file writer
            step (int): current training step.
        """
        with writer.as_default():
            tf.summary.scalar('total_loss', values[0], step=step)
            tf.summary.scalar('rpn_cls_loss', values[1], step=step)
            tf.summary.scalar('rpn_bbox_loss', values[2], step=step)
            tf.summary.scalar('head_cls_loss', values[3], step=step)
            tf.summary.scalar('head_bbox_loss', values[4], step=step)
            tf.summary.scalar('learning_rate', other_scalars[0], step=step)

    def write_bbox_img_summaries(self, figures, writer, step):
        """Add matplotlib figures into tensorboard.

        Args:
            figures (list): list of matplotlib figures to write.
            writer: a summary file writer
            step (int): current training step.
        """
        for fig in figures:
            # Format name if tensor.
            name = fig[1]
            if tf.is_tensor(name):
                name = tf.convert_to_tensor(name).numpy().decode()
            # Add figure to summary.
            self.add_image_to_summary(
                fig[0], writer, step, name=name,
                cmap=None, convert_as_fig=False,
                fig_size=(512, 512))

    @staticmethod
    def write_metric_summaries(values, names, writer, step):
        """Write metrics to tensorboard.

        Args:
            values (list): metric values to writes.
            names (list): metric names to writes.
            writer: a summary file writer
            step (int): current training step.
        """
        with writer.as_default():
            for name, val in zip(names, values):
                try:
                    tf.summary.scalar(str(name.numpy().decode('ascii')), val, step=step)
                except AttributeError:
                    # if run eagerly
                    tf.summary.scalar(str(name), val, step=step)

    @staticmethod
    def convert_to_tensor(tensors, data_type):
        """Convert list of tensors to tensor object.
        Use this function for non-eager mode."""

        def pad_tensor(tensor, max_dim):
            diff = max_dim - tf.shape(tensor)[0]
            if data_type == "bboxes":
                return tf.concat([tensor, tf.zeros((diff, 4), dtype=tf.int32)], axis=0)
            elif data_type == "labels":
                return tf.concat([tensor, tf.zeros((diff,), dtype=tf.int32)], axis=0)
            else:
                raise Exception("Data type not recognized.")

        # Pad tensors for common maximum dimension.
        dim_max = tf.reduce_max([tf.shape(item)[0] for item in tensors])
        return tf.stack([pad_tensor(tensor, dim_max) for tensor in tensors])

    def train(self, reset_summary_files=True):
        """Trains the learning model.

        Args:
            reset_summary_files (bool): if True, replace existing
                summary files. Relevant if a same model is re-trained.
        """
        # Delete previous summary event files from given folder.
        # Useful if training experiments require using same
        # summary output directories.
        if reset_summary_files:
            try:
                for directory in [self.summary_train_path, self.summary_validation_path]:
                    existing_summary_files = os.walk(directory).__next__()[-1]
                    if existing_summary_files:
                        for file in existing_summary_files:
                            os.remove(os.path.join(directory, file))
            except (PermissionError, StopIteration):
                pass

        # Create summary writers
        self.sum_train_writer = tf.summary.create_file_writer(self.summary_train_path)
        self.sum_validation_writer = tf.summary.create_file_writer(self.summary_validation_path)

        # Define saving increment for model parameters.
        save_increment = self.data_generator.data_size // config.NUM_SAVING_PER_EPOCH

        print('\n∎ Training')
        print('samples:', self.data_generator.data_size)
        print('augmentation:', config.AUGMENT)
        print('weights saving per epoch:', config.NUM_SAVING_PER_EPOCH)
        print(f'weights saved every {save_increment} steps', )
        print("model tracing...", end='\r')
        for epoch in range(config.EPOCHS):
            # Mean losses to display while training given
            # the summary iteration interval.
            avg_losses = tf.zeros(shape=(6,))

            for step in range(self.data_generator.data_size):
                start = time.time()
                print(f'step: {step}/{self.data_generator.data_size - 1} samples', end='\r')
                step = step + (epoch * self.data_generator.data_size)

                # Generate batch of training data.
                img, masks, bboxes, labels = self.data_generator.next_batch_with_cropping(
                    num_obj_thresh=[2, 50],
                    training=True,
                    data_aug=config.AUGMENT,
                    min_obj_size_ratio=0.3,
                    resize=True,
                    model_input_size=config.INPUT_SIZE)

                # Optimize and get loss.
                (loss_vector, boxes_prop,
                 metrics, metric_names) = self.learning_model.train_step(
                    model_args=[img,
                                tf.expand_dims(bboxes, axis=0),
                                tf.expand_dims(labels, axis=0),
                                tf.expand_dims(masks, axis=-1)],
                    optimize=tf.constant(True))
                avg_losses += loss_vector / config.SUMMARY_STEP

                # Get generated and ground truth boxes to display
                # as images in tensorboard.
                boxes_overview = self.get_box_overview_figure(
                    gt_bbox=bboxes,
                    pred_bbox=boxes_prop,
                    background=tf.zeros_like(img[0]))  # figure object

                self.write_loss_summaries(
                    values=loss_vector,
                    other_scalars=[self.learning_model.optimizer.lr(
                        self.learning_model.optimizer.iterations)],
                    writer=self.sum_train_writer,
                    step=tf.cast(step, tf.int64))

                self.write_bbox_img_summaries(
                    figures=boxes_overview,
                    writer=self.sum_train_writer,
                    step=tf.cast(step, tf.int64))

                self.write_metric_summaries(
                    values=metrics,
                    names=metric_names,
                    writer=self.sum_train_writer,
                    step=tf.cast(step, tf.int64))

                # Measure training loop execution time.
                end = time.time()
                speed = round(end - start, 2)

                # Display results at given interval.
                if (step % config.SUMMARY_STEP) == 0:
                    tf.print(f'epoch {epoch + 1}/{config.EPOCHS} ({step}, {speed} secs)' +
                             ' - total: {:.3f} - rpn_score: {:.3f}'
                             ' - rpn_bbox: {:.3f} - class: {:.3f}'
                             ' - bbox: {:.3f} - mask: {:.3f}'.format(*avg_losses))
                    # Reset average vector.
                    avg_losses = tf.zeros(shape=(6,))

                if step % config.VALIDATION_STEP == 0:
                    # Generate batch of validation data.
                    img, masks, bboxes, labels = self.data_generator.next_batch_with_cropping(
                        num_obj_thresh=[2, 50],
                        training=False,  # returns validation data if False
                        data_aug=config.AUGMENT,
                        min_obj_size_ratio=0.3,
                        resize=True,
                        model_input_size=(512, 512))

                    # Optimize and get loss.
                    loss_vector, _, metrics, metric_names = self.learning_model.train_step(
                        model_args=[img, tf.expand_dims(bboxes, axis=0), tf.expand_dims(labels, axis=0),
                                    tf.expand_dims(masks, axis=-1)],
                        optimize=tf.constant(False))

                    # Get regression and classification loss vector
                    # after optimizing.
                    self.write_loss_summaries(
                        values=loss_vector,
                        other_scalars=[self.learning_model.optimizer.lr(
                            self.learning_model.optimizer.iterations)],
                        writer=self.sum_validation_writer,
                        step=tf.cast(step, tf.int64))

                    self.write_metric_summaries(
                        values=metrics,
                        names=metric_names,
                        writer=self.sum_validation_writer,
                        step=tf.cast(step, tf.int64))

                # Do not save model if no saving increment per epoch.
                if save_increment == 0:
                    continue

                # Save model variables at given increment per epoch.
                if step % save_increment == 0:
                    # Define folder to solve weights.
                    basename = f'weights_epoch-{epoch}_step-{step}'
                    weight_folder_name = basename + datetime.now().strftime('_%Y%m%d-%H%M')
                    weight_folder_path = get_folder_or_create(
                        path=self.model_dir,
                        name=weight_folder_name)

                    # Assign learning rate to checkpoint.
                    current_lr = self.learning_model.optimizer.lr(
                        self.learning_model.optimizer.iterations)
                    self.ckpt.learning_rate.assign(
                        value=current_lr)

                    # Save model weights and checkpoint.
                    self.learning_model.save_variables(
                        directory=weight_folder_path,
                        checkpoint=self.ckpt)

    def test(self, input_ids=None):
        """Tests the learning model. Parameters of the model are
        assumed being restored prior to applying this method."""
        print("model tracing...", end='\r')

        data_size = len(self.data_generator.test_ids)
        metric_names = None
        stacked_losses = tf.TensorArray(tf.float32, size=data_size)
        stacked_metrics = tf.TensorArray(tf.float32, size=data_size)

        test_obj = self.data_generator.generate_test_data(input_ids)
        for i, data in enumerate(test_obj):
            start = time.time()
            # Define data
            img, masks, bboxes, labels = data

            # Call model and calculate losses.
            (losses, total_loss, _, metrics,
             metric_names) = self.learning_model.__call__(
                inputs=img,
                gt_boxes=tf.expand_dims(bboxes, axis=0),
                gt_cls=tf.expand_dims(labels, axis=0),
                gt_masks=tf.expand_dims(masks, axis=-1),
                is_training=False)

            # Measure training loop execution time
            end = time.time()
            speed = round(end - start, 2)

            # Display current losses.
            loss_vector = tf.concat([[total_loss], losses], axis=0)
            speed = f" ({speed} sec)"
            tf.print('total: {:.3f} - rpn_score: {:.3f}'
                     ' - rpn_bbox: {:.3f} - class: {:.3f}'
                     ' - bbox: {:.3f} - mask: {:.3f}'.format(*loss_vector) + speed)

            # Get average vectors.
            stacked_losses = stacked_losses.write(i, tf.constant(loss_vector))
            stacked_metrics = stacked_metrics.write(i, metrics)

        # Calculate average metrics.
        losses_val = stacked_losses.stack()
        values = tf.where(tf.math.is_nan(losses_val), tf.zeros_like(losses_val), losses_val)
        binary = tf.where(tf.math.is_nan(losses_val), tf.zeros_like(losses_val), tf.ones_like(losses_val))
        avg_losses = tf.reduce_sum(values, axis=0) / tf.reduce_sum(binary, axis=0)

        metrics_val = stacked_metrics.stack()
        values = tf.where(tf.math.is_nan(metrics_val), tf.zeros_like(metrics_val), metrics_val)
        binary = tf.where(tf.math.is_nan(metrics_val), tf.zeros_like(metrics_val), tf.ones_like(metrics_val))
        avg_metrics = tf.reduce_sum(values, axis=0) / tf.reduce_sum(binary, axis=0)

        tf.print('')
        names = ["rpn_score_loss", "rpn_bbx_loss",
                 "head_cls_loss", "head_bbox_loss",
                 "mask_loss"]
        for val, name in zip(avg_losses, names):
            met_results = "{}: {:.3f}".format(name, val)
            tf.print(met_results)

        try:
            # Graph mode.
            convert_name = lambda x: str(x.numpy().decode('ascii'))
            metric_names = [convert_name(name) for name in metric_names]
        except AttributeError:
            # Eagerly mode.
            convert_name = lambda x: str(x)
            metric_names = [convert_name(name) for name in metric_names]

        for val, name in zip(avg_metrics, metric_names):
            met_results = "{}: {:.3f}".format(name, val)
            tf.print(met_results)
