#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:08:19 2019

@author: niceliu
"""

import cv2
img=cv2.imread('/mnt/software/datasets/supervisely-person-datasets/ds10/img/pexels-photo-864895.png')
bbox_int=[309, 330, 613, 567]
c1 = (bbox_int[1], bbox_int[0])
c2 = (bbox_int[3], bbox_int[2])
cv2.rectangle(img,c1,c2,(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)