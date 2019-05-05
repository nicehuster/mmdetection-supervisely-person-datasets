# coding: utf-8
import supervisely_lib as sly
import os

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)

# get some context - team and workspace
team = api.team.get_list()[0]
workspace = api.workspace.get_list(team.id)[0]
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))


# clone from explore
model_name = 'yolo_coco'
if not api.model.exists(workspace.id, model_name):
    task_id = api.model.clone_from_explore('Supervisely/Model Zoo/YOLO v3 (COCO)', workspace.id, model_name)
    api.task.wait(task_id, api.task.Status.FINISHED)
model = api.model.get_info_by_name(workspace.id, model_name)
print("Model has been sucessfully cloned from explore: ")
print(model)


# get all neural networks in selected workspace
models = api.model.get_list(workspace.id)
print("Workspace {!r} contains {} models:".format(workspace.name, len(models)))
for model in models:
    print("{:<5}{:<15s}".format(model.id, model.name))


# get model info by name
model = api.model.get_info_by_name(workspace.id, model_name)
if model is None:
    print("Model {!r} not found".format(model_name))
else:
    print(model)

# get model info by id
some_team_id = model.id
model = api.model.get_info_by_id(some_team_id)
if model is None:
    print("Model with id={!r} not found".format(some_team_id))
else:
    print(model)
