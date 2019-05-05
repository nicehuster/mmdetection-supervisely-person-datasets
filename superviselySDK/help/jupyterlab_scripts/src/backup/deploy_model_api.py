# coding: utf-8
address = 'http://192.168.1.69:5555'
token = 'OfaV5z24gEQ7ikv2DiVdYu1CXZhMavU7POtJw2iDtQtvGUux31DUyWTXW6mZ0wd3IRuXTNtMFS9pCggewQWRcqSTUi4EJXzly8kH7MJL1hm3uZeM2MCn5HaoEYwXejKT'
team_name = 'max'
workspace_name = 'script2'
agent_name = 'max_pycharm'
model_name = 'yolo_pretrained'

#address = 'http://192.168.1.37'
#token = 'fpi35729nz8Hzcn8IifX7927HuYFKk99wzS3BAy5jkJZ5bFvOvziAXudrOQIjjCIBXAVgOySoOeeyMGarNQ6dkfOpKp7oyJO37tf9CO6NJU7Hn8RvMZV7jj0iWwQ4JdH'
#team_name = 'admin'
#workspace_name = 'Root'
#model_name = 'YOLO v3 (COCO)'
#agent_name = 'max_pycharm'

img_url = 'http://192.168.1.69:5555/h5un6l2bnaz1vj8a9qgms4-public/assets/projects/images/V/N/iQ/wOqv967pfMoMcJQ5Zo8j666wkABscaSR0f1N8lfKgc1eG98GVI1qxTv0UPiKsbsTFuCSEVhJpU6tCaYkUD0eJo69xzy3dwCRnvHaAwKFwZdr0jwfYrqQn0uf6PeN.png'

import requests
import numpy as np
import supervisely_lib as sly
import time

response = requests.get(img_url)
img = sly.image.read_bytes(response.content)
print(img.shape)

#@TODO: what if model has already deployed ???
#@TODO: why do we need workspace_id to deploy model???


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

responce = api.model.inference(model_info['id'], img)


x = 10