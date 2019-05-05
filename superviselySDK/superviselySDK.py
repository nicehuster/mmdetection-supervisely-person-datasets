#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:21:52 2019

@author: niceliu
"""

import numpy as np
import matplotlib.pyplot as plt

import supervisely_lib as sly  # Supervisely Python SDK

project = sly.Project('/mnt/software/datasets/supervisely-person-datasets/', sly.OpenMode.READ)


# Locate and load image labeling data.
item_paths = project.datasets.get('ds10').get_item_paths('pexels-photo-864895.png')

ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
# Go over the labeled objects and print out basic properties.
height,width=ann.img_size


def display_images(images, figsize=None):
    plt.figure(figsize=(figsize if (figsize is not None) else (15, 15)))
    for i, img in enumerate(images, start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)


# Load the original image too.
img = sly.image.read(item_paths.img_path)



rendered_bboxes = np.zeros(ann.img_size + (3,), dtype=np.uint8)

rendered_bboxes_test = np.zeros(ann.img_size + (3,), dtype=np.uint8)
labelx=ann.labels[0]
labelx.geometry.draw(rendered_bboxes_test, color=[1 for x in labelx.obj_class.color])
x=rendered_bboxes_test[:,:,2]

#bbox = labelx.geometry.to_bbox()
#bbox.draw_contour(rendered_bboxes_test, color = labelx.obj_class.color, thickness=20)

for label in ann.labels:
    print('Label type: ' + label.geometry.geometry_name())
    
    # Same call for any label type.
    bbox = label.geometry.to_bbox()
    print('Label bounding box: [{}, {}, {}, {}]'.format(
        bbox.top, bbox.left, bbox.bottom, bbox.right))
    print(label.obj_class.color)
    # Draw the bounding boxes.
    bbox.draw_contour(rendered_bboxes, color = label.obj_class.color, thickness=20)
    
    # Draw the labels themselves too to make sure the bounding boxes are correct.
    label.geometry.draw(rendered_bboxes, color=[int(x/2) for x in label.obj_class.color])

display_images([img,x])









