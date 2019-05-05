# coding: utf-8
import supervisely_lib as sly
import os
import subprocess

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

project_dir = '/workdir/downloaded_project'
sly.fs.remove_dir(project_dir)
project_fs = sly.Project(project_dir, sly.OpenMode.CREATE)

meta_json = api.project.get_meta(project.id)
meta = sly.ProjectMeta.from_json(meta_json)
project_fs.set_meta(meta)

for dataset in api.dataset.get_list(project.id):
    dataset_fs = project_fs.create_dataset(dataset.name)
    for image in api.image.get_list(dataset.id):
        ann_info = api.annotation.download(image.id)
        ann = sly.Annotation.from_json(ann_info.annotation, meta)
        img = api.image.download_np(image.id)
        dataset_fs.add_item_np(image.name, img, image.ext, ann)

print(subprocess.Popen("tree {}".format(project_dir), shell=True, stdout=subprocess.PIPE).stdout.read().decode('utf-8'))