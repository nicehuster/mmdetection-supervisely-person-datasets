# PyTorch Hello World neural net integration with Supervisely.

This example shows how to leverage Supervisely Python SDK to integrate a custom model into Supervisely framework. We
start with a basic PyTorch model and add all the wrappers necessary to run training and inference from Supervisely UI.

---

### Architecture

The network consists of 3 convolutional layers, each with kernel size 3. The number of channels is 3 (RGB input) - 10 - 20 - (number of output classes).


### Train configuration
Default train configuration is available in model presets (`predefined_run_configs.json` file). Those presets will be
automatically exposed in the UI. 

Common typicaly used parameters include:

- `lr` - Learning rate.
- `epochs` - the count of training epochs.
- `val_every` - validation peroid by epoch (value `0.5` mean 2 validations per epoch).
- `batch_size` - batch sizes for training (`train`) and validation (`val`) stages.
- `dataset_tags` - mapping for split data to train (`train`) and validation (`val`) parts by images tags. Images must
be tagged by `train` or `val` tags.
- `weights_init_type` - can be in one of 2 modes. In `transfer_learning` mode all possible weights will be transfered except last layer. In `continue_training` mode all weights will be transfered and validation for classes number and classes names order will be performed.

Full training configuration example:
```json
{
  "lr": 0.001,
  "epochs": 3,
  "val_every": 0.5,
  "batch_size": {
    "val": 12,
    "train": 12
  },
  "input_size": {
    "width": 256,
    "height": 256
  },
  "dataset_tags": {
    "val": "val",
    "train": "train"
  },
  "weights_init_type": "transfer_learning"
}
```

### Inference configuration

- `model` - model-specific config. For example, for a detection mode one may have a score threshold to filter detected
objects. Our example model does not have any special config, so this section is empty. 
- `mode` - one of inference modes that define the postprocessing logic: full image, ROI, objects. For explanation see [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config).

Full inference configuration example:

```json
{
    "model": {
    },
    "mode": {
    "name": "full_image",
    "model_classes": {
      "save_classes": "__all__",
      "add_suffix": "_pytorch_segm_example"
    }
  }
}
```