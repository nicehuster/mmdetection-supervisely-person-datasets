# coding: utf-8

# Supervisely imports.
import supervisely_lib as sly
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage
from supervisely_lib.nn.pytorch.weights import WeightsRW
from supervisely_lib.nn import raw_to_labels

# Third-party imports.
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms.functional import to_tensor

# Local imports.
from model import PyTorchSegmentation


class PytorchSegmentationSingleImageApplier(SingleImageInferenceBase):
    def _load_train_config(self):
        # Override config loading to load model-specific config (input size) in addition to the base config (mapping
        # between segmentation classes and integer IDs produced by the model)
        super()._load_train_config()
        self._determine_model_input_size()

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()   # Progress reporting done by the base class.
        # Check the class index --> name mapping to infer the number of model output dimensions.
        num_classes = max(self.out_class_mapping.keys()) + 1

        # Initialize the model.
        self._model = PyTorchSegmentation(num_classes=num_classes)
        sly.logger.info('Model has been instantiated.')

        # Load model weights.
        WeightsRW(sly.TaskPaths.MODEL_DIR).load_strictly(self._model)

        # Switch the model into evaluation mode (disable gradients computation and batchnorm updates).
        self._model.eval()

        # Move the model to the GPU.
        self._model.cuda()
        sly.logger.info('Model weights have been loaded.')

    def inference(self, img, ann):
        # Resize the image to the dimenstions the model has been trained on.
        # Even though the model preserves input dimensionality and in principle can be used with the full-size original
        # input directly, it is better to resize to preserve the scale of image features consistent with what the model
        # has seen during training.
        img_resized = sly.image.resize(img, self.input_size)

        # Height x Width x Channels -> Channels x Height x Width (pytorch convention), conversion to float,
        # change intensity range from 0...255 to 0...1
        img_chw = to_tensor(img_resized)

        # Add a dummy batch dimension.
        img_bchw = img_chw[None, ...]

        # Copy the data to the GPU to feed the model for inference.
        with torch.no_grad():
            model_input_cuda = Variable(img_bchw).cuda()

        # Run inference.
        class_scores_cuda = self._model(model_input_cuda)

        # Copy inference results back to CPU RAM, drop the batch dimension.
        class_scores = np.squeeze(class_scores_cuda.data.cpu().numpy(), axis=0)

        # Reorder dimensions back to the common Height x Width x Channels format.
        class_scores_hwc = np.transpose(class_scores, [1, 2, 0])

        # Resize back to the original size. Use nearest neighbor interpolation to avoid interpolation artifacts in the
        # scores space.
        class_scores_original_size = sly.image.resize_inter_nearest(class_scores_hwc, out_size=img.shape[:2])

        # Find the most likely class ID for every pixel.
        class_ids = np.argmax(class_scores_original_size, axis=2)

        # Convert the raw segmentation array to supervisely labels, one per output class.
        labels = raw_to_labels.segmentation_array_to_sly_bitmaps(self.out_class_mapping, class_ids)

        # Wrap the resulting labels into an image annotation and return.
        return sly.Annotation(img_size=img.shape[:2], labels=labels)


def main():
    # Bring up the model for inference (process class mapping configs; load model weights to the GPU).
    single_image_applier = PytorchSegmentationSingleImageApplier()
    # By default simply use the full image as model input. Other inference modes are possible, see
    # supervisely_lib/nn/hosted/inference_modes.py
    default_inference_mode_config = InfModeFullImage.make_default_config(
        model_result_suffix='_pytorch_segm_example')
    # IO wrapper to read inputs and save results in supervisely format within the context of a supervisely
    # agent task.
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config)
    # Process the input images and write out results.
    dataset_applier.run_inference()


if __name__ == '__main__':
    sly.main_wrapper('PYTORCH_SEGMENTATION_INFERENCE', main)
