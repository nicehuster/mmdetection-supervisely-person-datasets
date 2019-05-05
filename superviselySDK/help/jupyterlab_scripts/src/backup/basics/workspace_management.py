# coding: utf-8
import supervisely_lib as sly
import os

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)

# get some team
team = api.team.get_list()[0]
print("Current team: id = {}, name = {!r}".format(team.id, team.name))

# get all workspaces in selected team
workspaces = api.workspace.get_list(team_id=team.id)
print("Team {!r} contains {} workspaces:".format(team.name, len(workspaces)))
for workspace in workspaces:
    print("{:<5}{:<15s}".format(workspace.id, workspace.name))

# access WorkspaceInfo fields
workspace = workspaces[0]
print("Workspace information:")
print(workspace)

# create new workspace
workspace_name = 'test_workspace'
if api.workspace.exists(team.id, workspace_name):
    workspace_name = api.workspace.get_free_name(team.id, workspace_name)
created_workspace = api.workspace.create(team.id, workspace_name, 'test ws description')
print(created_workspace)


# get workspace info by name
workspace_name = 'test_workspace'
workspace = api.workspace.get_info_by_name(team.id, workspace_name)
if workspace is None:
    print("Workspace {!r} not found".format(workspace_name))
else:
    print(workspace)


# get workspace info by id
some_workspace_id = api.workspace.get_list(team.id)[0].id
workspace = api.team.get_info_by_id(some_workspace_id)
if workspace is None:
    print("Workspace with id={!r} not found".format(some_workspace_id))
else:
    print(workspace)


# update workspace name, description, or both
new_name = 'my_super_workspace'
new_description = 'super workspace description'
if api.workspace.exists(team.id, new_name):
    new_name = api.workspace.get_free_name(team.id, new_name)
updated_workspace = api.workspace.update(workspace.id, new_name, new_description)
print("Before update: {}".format(workspace))
print("After  update: {}".format(updated_workspace))
