# coding: utf-8
import supervisely_lib as sly
import os


team_name = 'max'
workspace_name = 'test_dtl_segmentation'
agent_name = 'max_pycharm' # None
init_model_name = 'nn_01'
input_project_name = 'lemons_test'
result_project_name = 'inf_test_api'


inference_config = {
  "mode": {
    "name": "full_image",
    "model_classes": {
      "add_suffix": "_model",
      "save_classes": "__all__"
    }
  },
  "model": {
    "gpu_device": 0
  }
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


task_id = api.task.run_inference(agent.id, project.id, model.id, result_project_name, inference_config)
print('Inference task (id={}) is started'.format(task_id))