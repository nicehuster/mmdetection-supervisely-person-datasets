import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/supervisely.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '/path/to/maskrcnn_supervisely_epoch_80-01b30339.pth')

# test a single image
img = mmcv.imread('/path/to/img.jpg')
result = inference_detector(model, img, cfg)
show_result(img, result,'supervisely',0.7,'pretrained_supervise_res.jpg')


