# coding: utf-8
import supervisely_lib as sly
import os
import requests
import numpy as np

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)


# get some context - team and workspace
team = api.team.get_list()[0]
workspace = api.workspace.get_list(team.id)[0]
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))

project_name = 'project_test'
if api.project.exists(workspace.id, project_name):
    project_name = api.project.get_free_name(workspace.id, project_name)
task_id = api.project.clone_from_explore('Supervisely/Demo/lemons_annotated', workspace.id, project_name)
api.task.wait(task_id, api.task.Status.FINISHED)
project = api.project.get_info_by_name(workspace.id, project_name)
print('Project {!r} has been sucessfully created: '.format(project.name))
print(project)
print('Number of images in project: ', api.project.get_images_count(project.id))

img_url = 'http://192.168.1.69:5555/h5un6l2bnaz1vj8a9qgms4-public/assets/projects/images/V/N/iQ/wOqv967pfMoMcJQ5Zo8j666wkABscaSR0f1N8lfKgc1eG98GVI1qxTv0UPiKsbsTFuCSEVhJpU6tCaYkUD0eJo69xzy3dwCRnvHaAwKFwZdr0jwfYrqQn0uf6PeN.png'
response = requests.get(img_url)
img = sly.image.read_bytes(response.content)
#visualize image here
print(img.shape)


new_dataset = api.dataset.create(project.id, 'new_dataset')
img_hash = api.image.upload_np(img)
img_info = api.image.add(new_dataset.id, 'super_image', img_hash)
print('image has been sucessfully added: id={}, name={}'.format(img_info.id, img_info.name))
print('Number of images in project: ', api.project.get_images_count(project.id))