# coding: utf-8
address = 'http://192.168.1.69:5555'
token = 'OfaV5z24gEQ7ikv2DiVdYu1CXZhMavU7POtJw2iDtQtvGUux31DUyWTXW6mZ0wd3IRuXTNtMFS9pCggewQWRcqSTUi4EJXzly8kH7MJL1hm3uZeM2MCn5HaoEYwXejKT'
team_name = 'max'
workspace_name = 'script2'

src_projects = ['teasm1/ws1/lemons_annotated', 'teasm3/ws222/roads_annotated', 'persons']
dst_project_name = 'merge_projects_01'


import supervisely_lib as sly
from tqdm import tqdm

api = sly.Api(address, token)

team_id = api.team.get_id_by_name(team_name)
workspace_id = api.workspace.get_id_by_name(workspace_name, team_id)


dst_meta = sly.ProjectMeta()
for src_project_name in src_projects:
    src_project_id = api.project.get_id_by_name(src_project_name, workspace_id)
    src_meta_json = api.project.get_meta(src_project_id)
    src_meta = sly.ProjectMeta.from_json(src_meta_json)
    dst_meta = dst_meta.merge(src_meta)

print(dst_meta)

dst_project_name = api.project.get_free_name(dst_project_name, workspace_id)
dst_project_info = api.project.create(dst_project_name, workspace_id)
dst_project_id = dst_project_info[api.project.ID]
api.project.update_meta(dst_project_id, dst_meta.to_json())


for src_project_name in src_projects:
    src_project_id = api.project.get_id_by_name(src_project_name, workspace_id)
    for dataset_info in api.dataset.get_list(src_project_id):
        src_dataset_id = dataset_info[api.dataset.ID]
        src_dataset_name = dataset_info[api.dataset.NAME]

        print('Project/Dataset: {}/{}'.format(src_project_name, src_dataset_name))

        dst_dataset_name = api.dataset.get_free_name(src_dataset_name, dst_project_id)
        dst_dataset_info = api.dataset.create(dst_dataset_name, dst_project_id)
        dst_dataset_id = dst_dataset_info[api.dataset.ID]

        for src_image_info in tqdm(api.image.get_list(src_dataset_id)):
            dst_img_name = api.image.get_free_name(src_image_info[api.image.NAME], dst_dataset_id)
            dst_img_id = api.image.add(dst_img_name, src_image_info[api.image.HASH], dst_dataset_id)[api.image.ID]
            src_ann = api.annotation.download(src_dataset_id, src_image_info[api.image.ID])
            api.annotation.upload(src_ann, dst_img_id)
        print()
