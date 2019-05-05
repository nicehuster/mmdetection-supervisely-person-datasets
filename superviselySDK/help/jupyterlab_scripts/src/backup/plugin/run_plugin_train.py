# coding: utf-8
import supervisely_lib as sly
import os


team_name = 'max'
workspace_name = 'test_dtl_segmentation'
agent_name = 'max_pycharm' # None
init_model_name = 'UNet (VGG weights)'
input_project_name = 'lemons_annotated_segmentation_013'
result_model_name = 'nn_01'


training_config = {
  "lr": 0.001,
  "epochs": 3,
  "val_every": 0.5,
  "batch_size": {
    "val": 6,
    "train": 12
  },
  "input_size": {
    "width": 256,
    "height": 256
  },
  "gpu_devices": [
    0
  ],
  "data_workers": {
    "val": 0,
    "train": 3
  },
  "dataset_tags": {
    "val": "val",
    "train": "train"
  },
  "special_classes": {
    "neutral": "neutral",
    "background": "bg"
  },
  "weights_init_type": "transfer_learning"
}


address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)

team = api.team.get_info_by_name(team_name)
workspace = api.workspace.get_info_by_name(team.id, workspace_name)
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))


model = api.model.get_info_by_name(workspace.id, init_model_name)
if model is None:
    raise RuntimeError("Model {!r} not found".format(init_model_name))

project = api.project.get_info_by_name(workspace.id, input_project_name)
if project is None:
    raise RuntimeError("Project {!r} not found".format(input_project_name))

agent = api.agent.get_info_by_name(team.id, agent_name)
if agent is None:
    raise RuntimeError("Agent {!r} not found".format(agent_name))
if agent.status is api.agent.Status.WAITING:
    raise RuntimeError("Agent {!r} is not running".format(agent_name))


task_id = api.task.run_train(agent.id, project.id, model.id, result_model_name, training_config)
print('Train task (id={}) is started'.format(task_id))