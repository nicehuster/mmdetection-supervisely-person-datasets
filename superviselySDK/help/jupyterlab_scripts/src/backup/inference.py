# coding: utf-8
address = 'http://192.168.1.69:5555'
token = 'OfaV5z24gEQ7ikv2DiVdYu1CXZhMavU7POtJw2iDtQtvGUux31DUyWTXW6mZ0wd3IRuXTNtMFS9pCggewQWRcqSTUi4EJXzly8kH7MJL1hm3uZeM2MCn5HaoEYwXejKT'
team_name = 'max'
workspace_name = 'script2'
agent_name = 'max_pycharm'
model_name = 'yolo_pretrained'
src_project_name = 'persons'
dst_project_name = 'persons_inf_yolo'

inference_settings = {
  "mode": {
    "source": "full_image"
  },
  "gpu_device": 0,
  "model_classes": {
    "add_suffix": "_yolo",
    "save_classes": "__all__"
  },
  "confidence_tag_name": "confidence"
}

import requests
import numpy as np
import supervisely_lib as sly
import time

api = sly.Api(address, token)
team_id = api.team.get_id_by_name(team_name)
workspace_id = api.workspace.get_id_by_name(workspace_name, team_id)
agent_id = api.agent.get_id_by_name(agent_name, team_id)

model_info = api.model.get_info_by_name(model_name, workspace_id)
plugin_id = model_info['pluginId']

plugin_info = api.plugin.get_info_by_id(plugin_id, team_id)
plugin_version = plugin_info['defaultVersion']

#@TODO: add option to send only image_hash or image_id, not image data

tasks_ids = api.model.get_deploy_tasks(model_info['id'])
if len(tasks_ids) == 0:
    task_id = api.task.deploy_model(agent_id, model_info['id'], workspace_id, 'never', {}, plugin_id, plugin_version)['taskId']
else:
    task_id = tasks_ids[0]

while True:
    status = api.task.get_status(task_id)
    api.task.raise_for_status(status)
    if status is api.task.Status.DEPLOYED:
        break
    time.sleep(2)





x = 10