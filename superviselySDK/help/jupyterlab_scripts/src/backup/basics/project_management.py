# coding: utf-8
import supervisely_lib as sly
import os
import json

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)


# get some context - team and workspace
team = api.team.get_list()[0]
workspace = api.workspace.get_list(team.id)[0]
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))


# clone project from explore
project_name = 'lemons_annotated'
if api.project.exists(team.id, project_name):
    project_name = api.project.get_free_name(workspace.id, project_name)
task_id = api.project.clone_from_explore('Supervisely/Demo/lemons_annotated', workspace.id, project_name)
api.task.wait(task_id, api.task.Status.FINISHED)
project = api.project.get_info_by_name(workspace.id, project_name)
print("Project has been sucessfully cloned from explore: ")
print(project)


# get all projects in selected workspace
projects = api.project.get_list(workspace.id)
print("Workspace {!r} contains {} projects:".format(workspace.name, len(projects)))
for project in projects:
    print("{:<5}{:<15s}".format(project.id, project.name))


# get project info by name
project = api.project.get_info_by_name(workspace.id, project_name)
if project is None:
    print("Workspace {!r} not found".format(project_name))
else:
    print(project)

# get project info by id
some_project_id = project.id
project = api.project.get_info_by_id(some_project_id)
if project is None:
    print("Project with id={!r} not found".format(some_project_id))
else:
    print(project)

# access ProjectInfo fields
print("Project information:")
print(project)


# get number of datasets and images in project
datasets_count = api.project.get_datasets_count(project.id)
images_count = api.project.get_images_count(project.id)
print("Project {!r} contains:\n {} datasets \n {} images\n".format(project.name, datasets_count, images_count))


# get project meta information, that contains list of defined object classes, image tags, object tags
meta_json = api.project.get_meta(project.id)
print(json.dumps(meta_json, indent=4))

#convert json to ProjectMeta object
meta = sly.ProjectMeta.from_json(meta_json)
print(meta)

# get list of datasets
datasets = api.dataset.get_list(project.id)
print("Project {!r} contains {} datasets:".format(project.name, len(datasets)))
for dataset in datasets:
    print("Id: {:<5} Name: {:<15s} ImagesCount: {:<5}".format(dataset.id, dataset.name, dataset.images_count))


# iterate over images in dataset
dataset = datasets[0]
images = api.image.get_list(dataset.id)
print("Dataset {!r} contains {} images:".format(dataset.name, len(images)))
for image in images:
    print("Id: {:<5} Name: {:<15s} LabelsCount: {:<5} Size(bytes): {:<10} Width: {:<5} Height: {:<5}"
          .format(image.id, image.name, image.labels_count, image.size, image.width, image.height))

#download image and its annotation
image = images[0]
img = api.image.download_np(image.id)
print("Image Shape: {}".format(img.shape))
#visualize image here

ann_info = api.annotation.download(image.id)
print(json.dumps(ann_info.annotation, indent=4))

ann = sly.Annotation.from_json(ann_info.annotation, meta)
#draw img + ann
