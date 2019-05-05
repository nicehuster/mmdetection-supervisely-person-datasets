# coding: utf-8
import supervisely_lib as sly
import os
import matplotlib.pyplot as plt
import numpy as np

team_name = 'max'
workspace_name = 'test_dtl_segmentation'
agent_name = 'max_pycharm' # None
model_name = 'yolo_coco'

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)

team = api.team.get_info_by_name(team_name)
workspace = api.workspace.get_info_by_name(team.id, workspace_name)
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))


if not api.model.exists(workspace.id, model_name):
    task_id = api.model.clone_from_explore('Supervisely/Model Zoo/YOLO v3 (COCO)', workspace.id, model_name)
    api.task.wait(task_id, api.task.Status.FINISHED)
model = api.model.get_info_by_name(workspace.id, model_name)
print(model)

#demo image
img_path = '/workdir/demo_data/images/friends.jpeg'
img = sly.image.read(img_path)


agent = api.agent.get_info_by_name(team.id, agent_name)
if agent is None:
    raise RuntimeError("Agent {!r} not found".format(agent_name))
if agent.status is api.agent.Status.WAITING:
    raise RuntimeError("Agent {!r} is not running".format(agent_name))


task_ids = api.model.get_deploy_tasks(model.id)
if len(task_ids) == 0:
    print('Model {!r} is not deployed. Deploying...'.format(model.name))
    task_id = api.task.deploy_model(agent.id, model.id)
    api.task.wait(task_id, api.task.Status.DEPLOYED)
else:
    print('Model {!r} has been already deployed'.format(model.name))
    task_id = task_ids[0]

print('Deploy task_id = {}'.format(task_id))


output_meta_json = api.model.get_output_meta(model.id)
output_meta = sly.ProjectMeta.from_json(output_meta_json)
print(output_meta)


ann_json = api.model.inference(model.id, img)
ann = sly.Annotation.from_json(ann_json, output_meta)
print('Model has been sucessfully applied to the image')


#draw_img = np.copy(img)
#ann.draw(draw_img, thickness=10)
#plt.figure()
#plt.imshow(draw_img)
