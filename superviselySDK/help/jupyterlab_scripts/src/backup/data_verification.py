# coding: utf-8
import numpy as np

from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.annotation.label import Label
from supervisely_lib.annotation.tag import Tag
from supervisely_lib.geometry.bitmap import Bitmap
from supervisely_lib.project.project_meta import ProjectMeta


def make_false_positive_name(cls_name):
    return cls_name + '_FP'


def make_false_negative_name(cls_name):
    return cls_name + '_FN'


def make_iou_tag_name(cls_name):
    return cls_name + '_iou'


def get_added_tags(self):
    tags = []
    for first_class, _ in self.settings['classes_matching'].items():
        tag_name = self._make_iou_tag_name(first_class)
        tags.append({'name': tag_name, 'value': 'any'})
    return tags


def _compute_masks_iou(mask_1, mask_2):
    if mask_1.sum() == 0 and mask_2.sum() == 0:
        return 1.0
    intersection = (mask_1 & mask_2).sum()
    union = mask_1.sum() + mask_2.sum() - intersection
    return intersection / union


def _create_fp_mask(mask_gt, mask_pred):
    fp_mask = mask_pred.copy()
    fp_mask[mask_gt] = False
    return fp_mask


def _create_fn_mask(mask_gt, mask_pred):
    fn_mask = mask_gt.copy()
    fn_mask[mask_pred] = False
    return fn_mask


def verify_data(orig_ann: Annotation, classes_matching: dict, res_project_meta: ProjectMeta) -> Annotation:
    ann = orig_ann.clone()
    imsize = ann.img_size

    for first_class, second_class in classes_matching.items():
        mask1 = np.zeros(imsize, dtype=np.bool)
        mask2 = np.zeros(imsize, dtype=np.bool)
        for label in ann.labels:
            if label.obj_class.name == first_class:
                label.geometry.draw(mask1, True)
            elif label.obj_class.name == second_class:
                label.geometry.draw(mask2, True)

        iou_value = _compute_masks_iou(mask1, mask2)

        tag_meta = res_project_meta.img_tag_metas.get(make_iou_tag_name(first_class))
        tag = Tag(tag_meta, iou_value)
        ann.add_tag(tag)

        fp_mask = _create_fp_mask(mask1, mask2)
        if fp_mask.sum() != 0:
            fp_object_cls = res_project_meta.obj_classes.get(make_false_positive_name(first_class))
            fp_geom = Bitmap(data=fp_mask)
            fp_label = Label(fp_geom, fp_object_cls)
            ann.add_label(fp_label)

        fn_mask = _create_fn_mask(mask1, mask2)
        if fn_mask.sum() != 0:
            fn_object_cls = res_project_meta.obj_classes.get(make_false_negative_name(first_class))
            fn_geom = Bitmap(data=fn_mask)
            fn_label = Label(fn_geom, fn_object_cls)
            ann.add_label(fn_label)
    return ann
