import mmcv
import numpy as np
import torch
import pycocotools.mask as maskUtils
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.core import get_classes


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale=cfg.data.test.img_scale)
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)

import cv2
def plot_boxes(img,bboxes,labels,class_names,score_thr):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        c1 = (bbox_int[0], bbox_int[1])
        c2 = (bbox_int[2], bbox_int[3])
        color=colors[label]
        cv2.rectangle(img, c1, c2, color, 2)
        t1 = round(0.002 * max(img.shape[0:2])) + 1
        tf = max(t1 - 1, 1)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if label_text!='person': continue
        if len(bbox) > 4:
            label_text += '{:.02f}'.format(bbox[-1])

        t_size = cv2.getTextSize(label_text, 0, fontScale=t1 / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label_text, (c1[0], c1[1] - 2), 0, t1 / 4, [255, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)

    return img


def show_result(img, result, dataset='coco', score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    img_res=plot_boxes(img.copy(),bboxes,labels,
                       class_names=class_names,
                       score_thr=score_thr)

    if out_file:
        cv2.imwrite(out_file,img_res)

    '''

    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None)
    '''
