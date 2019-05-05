'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: supervisely.py
@time: 19-4-27 下午9:45
@desc:
'''
import mmcv,os
import numpy as np
from .custom import CustomDataset

import sys,tqdm
SuperviselySDK='/mnt/software/niceliu/mmdetection/superviselySDK'
sys.path.append(SuperviselySDK)
import supervisely_lib as sly  # Supervisely Python SDK


class SuperviselyDataset(CustomDataset):

    CLASSES = ('person')

    def load_annotations(self, ann_file):
        self.project = sly.Project(self.img_prefix, sly.OpenMode.READ)
        img_infos=list()
        anno_list=mmcv.list_from_file(ann_file)
        print('data loading ...')
        img_infos=mmcv.track_parallel_progress(self._load_ann,anno_list,16)
        print('data loading finished !!!')

        return img_infos

    def _load_ann(self,imgpath):
        item_paths = self.project.datasets.get(imgpath.split('/')[0]).get_item_paths(imgpath.split('/')[-1])
        anninfo = sly.Annotation.load_json_file(item_paths.ann_path, self.project.meta)
        height, width, ann = self._parse_ann_info(anninfo)
        return dict(filename=imgpath,width=width,height=height,ann=ann)

    def _parse_ann_info(self, ann_info, with_mask=True):

        height,width=ann_info.img_size

        gt_bboxes=[]
        gt_labels = []
        if with_mask:
            gt_masks = []

        for label in ann_info.labels:

            bbox = label.geometry.to_bbox()
            gt_bboxes.append([bbox.left,bbox.top, bbox.right,bbox.bottom])
            gt_labels.append(1)  # only person class
            render = np.zeros(ann_info.img_size + (3,), dtype=np.uint8)
            label.geometry.draw(render, color=[1 for x in label.obj_class.color])
            gt_masks.append(render[:,:,2])

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann=dict(bboxes=gt_bboxes, labels=gt_labels,bboxes_ignore=gt_bboxes_ignore)
        if with_mask:
            ann['masks'] = gt_masks

        return height,width,ann
