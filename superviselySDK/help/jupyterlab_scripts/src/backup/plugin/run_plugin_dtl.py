# coding: utf-8
import supervisely_lib as sly
import os
import json

team_name = 'max'
workspace_name = 'test_dtl_segmentation'
agent_name = 'max_pytharm' # None

src_project_name = 'lemons_annotated'
dst_project_name = 'lemons_annotated_segmentation'


path_dtl_graph = '/workdir/demo_data/dtl_segmentation_graph.json'
with open(path_dtl_graph, 'r') as file:
    dtl_graph_str = file.read()
dtl_graph_str = dtl_graph_str.replace('%SRC_PROJECT_NAME%', src_project_name)
dtl_graph_str = dtl_graph_str.replace('%DST_PROJECT_NAME%', dst_project_name)
dtl_graph = json.loads(dtl_graph_str)
print('DTL graph:')
print(json.dumps(dtl_graph, indent=4))


address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)

team = api.team.get_info_by_name(team_name)
workspace = api.workspace.get_info_by_name(team.id, workspace_name)
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))

agent = api.agent.get_info_by_name(team.id, agent_name)
if agent is not None and agent.status is api.agent.Status.WAITING:
    agent = None
agent_id = None if agent is None else agent.id


task_id = api.task.run_dtl(workspace.id, dtl_graph, agent_id)

print('DTL task (id={}) is started'.format(task_id))

#api.task.wait(task_id, api.task.Status.FINISHED)
#print('DTL task (id={}) has been successfully finished'.format(task_id))







