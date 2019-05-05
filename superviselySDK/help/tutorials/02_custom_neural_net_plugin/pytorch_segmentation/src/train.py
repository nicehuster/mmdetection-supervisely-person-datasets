# coding: utf-8

import supervisely_lib as sly
from supervisely_lib.nn.dataset import SlyDataset, ensure_samples_nonempty
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer, BATCH_SIZE, DATASET_TAGS, EPOCHS, LOSS, LR, \
    TRAIN, VAL, WEIGHTS_INIT_TYPE
from supervisely_lib.nn.pytorch.weights import WeightsRW
from supervisely_lib.nn.training.eval_planner import EvalPlanner, VAL_EVERY
from supervisely_lib.task.progress import epoch_float

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from model import PyTorchSegmentation, INPUT_SIZE, HEIGHT, WIDTH


class PytorchSlyDataset(SlyDataset):
    """Thin wrapper around the base Supervisely dataset IO logic to handle PyTorch specific conversions.

    The base class handles locating and reading imagea and annotation data from disk, conversions between named classes
    and integer class ids to feed to the models, and rendering annotations as images.
    """

    def _get_sample_impl(self, img_fpath, ann_fpath):
        # Read the image, read the annotatio, render the annotation as a bitmap of per-pixel class ids, resize to
        # requested model input size.
        img, gt = super()._get_sample_impl(img_fpath=img_fpath, ann_fpath=ann_fpath)

        # PyTorch specific logic:
        # - convert from (H x W x Channels) to (Channels x H x W);
        # - convert from uint8 to float32 data type;
        # - move from 0...255 to 0...1 intensity range.
        img_tensor = to_tensor(img)

        return img_tensor, gt


class PytorchSegmentationTrainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            INPUT_SIZE: {
                WIDTH: 256,
                HEIGHT: 256
            },
            DATASET_TAGS: {
                TRAIN: 'train',
                VAL: 'val',
            },
            BATCH_SIZE: {
                TRAIN: 1,
                VAL: 1,
            },
            EPOCHS: 2,
            VAL_EVERY: 0.5,
            LR: 0.001,
            WEIGHTS_INIT_TYPE: TRANSFER_LEARNING,  # CONTINUE_TRAINING,
        }

    def __init__(self):
        super().__init__(default_config=PytorchSegmentationTrainer.get_default_config())

    def get_start_class_id(self):
        """Set the integer segmentation class indices to start from 0.

        Some segmentation neural net implementations treat class id 0 in a special way (e.g. as a background class) and
        need the indexing to start from 1.
        """
        return 0

    def _determine_model_classes(self):
        """Look at input dataset segmentation classes and assign integer class ids."""

        # This also automatically reuses an existing class id mapping in continue_training mode (continuing training
        # from a previous snapshot with a dataset fully compatible in terms of classes).
        super()._determine_model_classes_segmentation(bkg_input_idx=-1)

    def _construct_and_fill_model(self):
        # Progress reporting to show a progress bar in the UI.
        model_build_progress = sly.Progress('Building model:', 1)

        # Check the class name --> index mapping to infer the number of model output dimensions.
        num_classes = max(self.class_title_to_idx.values()) + 1

        # Initialize the model.
        model = PyTorchSegmentation(num_classes=num_classes)
        sly.logger.info('Model has been instantiated.')

        # Load model weights appropriate for the given training mode.
        weights_rw = WeightsRW(sly.TaskPaths.MODEL_DIR)
        weights_init_type = self.config[WEIGHTS_INIT_TYPE]
        if weights_init_type == TRANSFER_LEARNING:
            # For transfer learning, do not attempt to load the weights for the model head. The existing snapshot may
            # have been trained on a different dataset, even on a different set of classes, and is in general not
            # compatible with the current model even in terms of dimensions. The head of the model will be initialized
            # randomly.
            self._model = weights_rw.load_for_transfer_learning(model, ignore_matching_layers=['_head'],
                                                                logger=sly.logger)
        elif weights_init_type == CONTINUE_TRAINING:
            # Continuing training from an older snapshot requires full compatibility between the two models, including
            # class index mapping. Hence the snapshot weights must exactly match the structure of our model instance.
            self._model = weights_rw.load_strictly(model)

        # Model weights have been loaded, move them over to the GPU.
        self._model.cuda()

        # Advance the progress bar and log a progress message.
        sly.logger.info('Weights have been loaded.', extra={WEIGHTS_INIT_TYPE: weights_init_type})
        model_build_progress.iter_done_report()

    def _construct_loss(self):
        # Initialize the logic for computing optimization loss.
        # Here one can also add other interesting metrics, e.g. accuracy. to be tracked.
        self._loss_fn = torch.nn.modules.loss.CrossEntropyLoss()

    def _construct_data_loaders(self):
        # Initialize the IO logic to feed the model during training.

        # Dimensionality of all images in an input batch must be the same.
        # We fix the input size for the whole dataset. Every image will be scaled to this size before feeding the model.
        src_size = self.config[INPUT_SIZE]
        input_size = (src_size[HEIGHT], src_size[WIDTH])

        # We need separate data loaders for the training and validation folds.
        self._data_loaders = {}

        # The train dataset should be re-shuffled every epoch, but the validation dataset samples order is fixed.
        for dataset_name, need_shuffle, drop_last in [(TRAIN, True, True), (VAL, False, False)]:
            # For more informative logging, grab the tag marking the respective dataset images.
            dataset_tag = self.config[DATASET_TAGS][dataset_name]

            # Get a list of samples for the dataset in question, make sure it is not empty.
            samples = self._samples_by_data_purpose[dataset_name]
            ensure_samples_nonempty(samples, dataset_tag, self.project.meta)

            # Instantiate the dataset object to handle sample indexing and image resizing.
            dataset = PytorchSlyDataset(
                project_meta=self.project.meta,
                samples=samples,
                out_size=input_size,
                class_mapping=self.class_title_to_idx,
                bkg_color=-1
            )
            # Report progress.
            sly.logger.info('Prepared dataset.', extra={
                'dataset_purpose': dataset_name, 'tag': dataset_tag, 'samples': len(samples)
            })
            # Initialize a PyTorch data loader. For the training dataset, set the loader to ignore the last incomplete
            # batch to avoid noisy gradients and batchnorm updates.
            self._data_loaders[dataset_name] = DataLoader(
                dataset=dataset,
                batch_size=self.config[BATCH_SIZE][dataset_name],
                shuffle=need_shuffle,
                drop_last=drop_last
            )

        # Report progress
        sly.logger.info('DataLoaders have been constructed.')

        # Compute the number of iterations per epoch for training and validation.
        self._train_iters = len(self._data_loaders[TRAIN])
        self._val_iters = len(self._data_loaders[VAL])
        self._epochs = self.config[EPOCHS]

        # Initialize a helper to determine when to pause training and perform validation and snapshotting.
        self._eval_planner = EvalPlanner(epochs=self._epochs, val_every=self.config[VAL_EVERY])

    def _dump_model_weights(self, out_dir):
        # Framework-specific logic to snapshot the model weights.
        WeightsRW(out_dir).save(self._model)

    def _validation(self):
        # Compute validation metrics.

        # Switch the model to evaluation model to stop batchnorm runnning average updates.
        self._model.eval()
        # Initialize the totals counters.
        validated_samples = 0
        total_loss = 0.0

        # Iterate over validation dataset batches.
        for val_it, (inputs, targets) in enumerate(self._data_loaders[VAL]):
            # Move the data to the GPU and run inference.
            with torch.no_grad():
                inputs_cuda, targets_cuda = Variable(inputs).cuda(), Variable(targets).cuda()
            outputs_cuda = self._model(inputs_cuda)

            # The last betch may be smaller than the rest if the dataset does not have a whole number of full batches,
            # so read the batch size from the input.
            batch_size = inputs_cuda.size(0)

            # Compute the loss and grab the value from GPU.
            loss_value = self._loss_fn(outputs_cuda, targets_cuda).item()

            # Add up the totals.
            total_loss += loss_value * batch_size
            validated_samples += batch_size

            # Report progress.
            sly.logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                             'val_iter': val_it, 'val_iters': self._val_iters})

        # Compute the average loss from the accumulated totals.
        metrics_values = {LOSS: total_loss / validated_samples}

        # Report progress and metric values to be plotted in the training chart and return.
        sly.report_metrics_validation(self.epoch_flt, metrics_values)
        sly.logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return metrics_values

    def train(self):
        # Initialize the progesss bar in the UI.
        training_progress = sly.Progress('Model training: ', self._epochs * self._train_iters)

        # Initialize the optimizer.
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config[LR])
        # Running best loss value to determine which snapshot is the best so far.
        best_val_loss = float('inf')

        for epoch in range(self._epochs):
            sly.logger.info("Starting new epoch", extra={'epoch': self.epoch_flt})
            for train_it, (inputs_cpu, targets_cpu) in enumerate(self._data_loaders[TRAIN]):
                # Switch the model into training mode to enable gradient backpropagation and batch norm running average
                # updates.
                self._model.train()

                # Copy input batch to the GPU, run inference and compute optimization loss.
                inputs_cuda, targets_cuda = Variable(inputs_cpu).cuda(), Variable(targets_cpu).cuda()
                outputs_cuda = self._model(inputs_cuda)
                loss = self._loss_fn(outputs_cuda, targets_cuda)

                # Make a gradient descent step.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Advance UI progess bar.
                training_progress.iter_done_report()
                # Compute fractional epoch value for more precise metrics reporting.
                self.epoch_flt = epoch_float(epoch, train_it + 1, self._train_iters)
                # Report metrics to be plotted in the training chart.
                sly.report_metrics_training(self.epoch_flt, {LOSS: loss.item()})

                # If needed, do validation and snapshotting.
                if self._eval_planner.need_validation(self.epoch_flt):
                    # Compute metrics on the validation dataset.
                    metrics_values_val = self._validation()

                    # Report progress.
                    self._eval_planner.validation_performed()

                    # Check whether the new weights are the best so far on the validation dataset.
                    val_loss = metrics_values_val[LOSS]
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss

                    # Save a snapshot with the current weights. Mark whether the snapshot is the best so far in terms of
                    # validation loss.
                    self._save_model_snapshot(model_is_best, opt_data={
                        'epoch': self.epoch_flt,
                        'val_metrics': metrics_values_val,
                    })

            # Report progress
            sly.logger.info("Epoch has finished", extra={'epoch': self.epoch_flt})


def main():
    # Read the training config and initialize all the training logic.
    x = PytorchSegmentationTrainer()
    # Run the training loop.
    x.train()


if __name__ == '__main__':
    sly.main_wrapper('UNET_V2_TRAIN', main)
