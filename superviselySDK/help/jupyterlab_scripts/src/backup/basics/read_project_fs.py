# coding: utf-8
import supervisely_lib as sly
import json

project_path = '/workdir/demo_data/lemons_annotated'

project = sly.Project(project_path, sly.OpenMode.READ)
print(project.meta)


dataset_names = project.datasets.keys()
print("Project {!r} consists of {} datasets: {}".format(project.name, len(dataset_names), dataset_names))

for dataset in project:
    for item_name in dataset:
        img_path, ann_path = dataset.get_item_paths(item_name)
        print("Current item {!r}".format(item_name))
        print("\t image path: {!r}".format(img_path))
        print("\t annotation path: {!r}".format(ann_path))

        img = sly.image.read(img_path)

        with open(ann_path) as json_data:
            ann_json = json.load(json_data)
        ann = sly.Annotation.from_json(ann_json, project.meta)

        ann.draw(img)
        #visualize image and annotation here

