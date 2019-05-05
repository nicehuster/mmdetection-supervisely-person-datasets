#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:21:52 2019

@author: niceliu
"""

import numpy as np
import matplotlib.pyplot as plt

import supervisely_lib as sly  # Supervisely Python SDK

# Open existing project on disk.
project = sly.Project('/mnt/software/datasets/supervisely-person-datasets/', sly.OpenMode.READ)
# Locate and load image labeling data.
item_paths = project.datasets.get('ds1').get_item_paths('pexels-photo-708392.png')
ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
# Go over the labeled objects and print out basic properties.
for label in ann.labels:
    print('Found label object: ' + label.obj_class.name)
    print('   geometry type: ' + label.geometry.geometry_name())
    print('   object area: ' + str(label.geometry.area))


# Read the underlying raw image for display.
img = sly.image.read(item_paths.img_path)
# Render the labeled objects.
ann_render = np.zeros(ann.img_size + (3,), dtype=np.uint8)
ann.draw(ann_render)
# Separately, render the labeled objects contours.
ann_contours = np.zeros(ann.img_size + (3,), dtype=np.uint8)
ann.draw_contour(ann_contours, thickness=7)
plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.imshow(ann_render)
plt.subplot(1, 3, 3)
plt.imshow(ann_contours)



##############################################

# Print basic project metadata.
print("Project name: ", project.name)
print("Project directory: ", project.directory)
print("Total images: ", project.total_items)
print("Dataset names: ", project.datasets.keys())
print("\n")

# What datasets and images are there and where are they on disk?

'''
for dataset in project:
    print("Dataset: ", dataset.name)

    # A dataset item is a pair of an image and its annotation.
    # The annotation contains all the labeling information about
    # the image - segmentation masks, objects bounding boxes etc.
    # We will look at annotations in detail shortly.
    for item_name in dataset:
        print(item_name)
        img_path = dataset.get_img_path(item_name)
        print("  image: ", img_path)
    print()
'''
##################################################


# Grab the file paths for both raw image and annotation in one call.
item_paths = project.datasets.get('ds1').get_item_paths('pexels-photo-708392.png')

# Load and deserialize annotation from JSON format.
# Annotation data is cross-checked again project meta, and references to
# the right LabelMeta and TagMeta objects are set up.
ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)

print('Loaded annotation has {} labels and {} image tags.'.format(len(ann.labels), len(ann.img_tags)))
print('Label class names: ' + (', '.join(label.obj_class.name for label in ann.labels)))
print('Image tags: ' + (', '.join(tag.get_compact_str() for tag in ann.img_tags)))
##################################################

# Basic imaging functionality and Jupyter image display helpers.
import numpy as np
from matplotlib import pyplot as plt
# A helper to display several images in a row.
# Can be safely skipped - not essentiall for understanding the rest of the code.
def display_images(images, figsize=None):
    plt.figure(figsize=(figsize if (figsize is not None) else (15, 15)))
    for i, img in enumerate(images, start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)

# Set up a 3-channel black canvas to render annotation labels on.
# Make the canvas size match the original image size.
ann_render = np.zeros(ann.img_size + (3,), dtype=np.uint8)

# Render all the labels using colors from the meta information.
ann.draw(ann_render)

# Set up canvas to draw label contours.
ann_contours = np.zeros(ann.img_size + (3,), dtype=np.uint8)

# Draw thick contours for the labels on a separate canvas.
ann.draw_contour(ann_contours, thickness=7)

# Load the original image too.
img = sly.image.read(item_paths.img_path)

# Display everything.
display_images([img, ann_render, ann_contours])


######################################Geometric labels



# Load the data and set up black canvas.
item_paths = project.datasets.get('ds1').get_item_paths('pexels-photo-708392.png')
ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
img = sly.image.read(item_paths.img_path)
rendered_labels = np.zeros(ann.img_size + (3,), dtype=np.uint8)

for label in ann.labels:
    print('Label type: ' + label.geometry.geometry_name())
    # Same call for any geometry type.
    label.draw(rendered_labels)

display_images([img, rendered_labels])

rendered_bboxes = np.zeros(ann.img_size + (3,), dtype=np.uint8)

for label in ann.labels:
    print('Label type: ' + label.geometry.geometry_name())
    
    # Same call for any label type.
    bbox = label.geometry.to_bbox()
    print('Label bounding box: [{}, {}, {}, {}]'.format(
        bbox.top, bbox.left, bbox.bottom, bbox.right))
    
    # Draw the bounding boxes.
    bbox.draw_contour(rendered_bboxes, color = label.obj_class.color, thickness=20)
    
    # Draw the labels themselves too to make sure the bounding boxes are correct.
    label.geometry.draw(rendered_bboxes, color=[int(x/2) for x in label.obj_class.color])

display_images([img, rendered_bboxes])


























