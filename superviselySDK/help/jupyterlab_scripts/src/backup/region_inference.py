# coding: utf-8
import time
from tqdm import tqdm
import supervisely_lib as sly

address = 'http://192.168.1.69:5555'
token = 'YGPDnuBkhFmcQ7VNzSEjhgavjg4eFR4Eq1C3jIY4HgV3SQq2JgkXCNtgZy1Fu2ftd4IKui8DsjrdtXjB853cMtBevpSJqFDYiaG1A5qphlH6fFiYYmcVZ5fMR8dDrt5l'
team_name = 'dima'
workspace_name = 'region'
agent_name = 'dima_agent'
models_settings = [{
        'name': 'stage01',
        'mode': {
            "save": False,
            "bounds": {
              "top": "50%",
              "left": "0%",
              "right": "00%",
              "bottom": "00%"
            },
            "name": "roi",
            "class_name": "inference_roi",
          "model_classes": {
            "add_suffix": "_model",
            "save_classes": ["road"]
          }}
    },
    {
        'name': 'stage02',
        'mode': {
            "save": False,
            "bounds": {
              "top": "50%",
              "left": "0%",
              "right": "0%",
              "bottom": "0%"
            },
            "name": "roi",
            "class_name": "inference_roi",
          "model_classes": {
            "add_suffix": "_model",
            "save_classes": [
              "crack"
            ]
          }
        }
    },
    {
        'name': 'stage03',
        'mode': {
            "save": False,
            "bounds": {
              "top": "50%",
              "left": "0px",
              "right": "0px",
              "bottom": "0px"
            },
            "name": "roi",
            "class_name": "inference_roi",
          "model_classes": {
            "add_suffix": "_model",
            "save_classes": [
              "detect_class"
            ]
          }
        }
    },
    {
        'name': 'stage04',
        'mode': {
            "save": False,
            "name": "bboxes",
            "padding": {
              "top": "15%",
              "left": "15%",
              "right": "15%",
              "bottom": "15%"
            },
            "add_suffix": "_input_bbox",
            "from_classes": [
              "detect_class_model"
            ],
          "model_classes": {
            "add_suffix": "_model",
            "save_classes": [
              "patch",
              "hole"
            ]
          }
        }
    }
]

src_project_name = 'road_test'
dst_project_name = 'res'


api = sly.Api(address, token)
team_id = api.team.get_id_by_name(team_name)
workspace_id = api.workspace.get_id_by_name(workspace_name, team_id)
agent_id = api.agent.get_id_by_name(agent_name, team_id)


def deploy_model(model_name):
    model_info = api.model.get_info_by_name(model_name, workspace_id)
    plugin_id = model_info['pluginId']

    plugin_info = api.plugin.get_info_by_id(plugin_id, team_id)
    plugin_version = plugin_info['defaultVersion']

    tasks_ids = api.model.get_deploy_tasks(model_info['id'])
    if len(tasks_ids) == 0:
        task_id = api.task.deploy_model(agent_id, model_info['id'], workspace_id, 'never', {},
                                        plugin_id, plugin_version)['taskId']
    else:
        task_id = tasks_ids[0]

    while True:
        status = api.task.get_status(task_id)
        api.task.raise_for_status(status)
        if status is api.task.Status.DEPLOYED:
            break
        time.sleep(2)

    return model_info


# Deploy all model
for m_s in models_settings:
    print('Deploying - {}'.format(m_s['name']))
    model_info = deploy_model(m_s['name'])
    m_s['info'] = model_info


dst_project_name = api.project.get_free_name(dst_project_name, workspace_id)
dst_project_id = api.project.create(dst_project_name, workspace_id)['id']

src_project_id = api.project.get_info_by_name(src_project_name, workspace_id)['id']

src_meta_json = api.project.get_meta(src_project_id)
src_meta = sly.ProjectMeta.from_json(src_meta_json)

dst_meta = src_meta.clone()

model_src_meta = src_meta.to_json()
for m_s in models_settings:
    m_s['src_meta'] = model_src_meta
    model_meta_json = api.model.get_project_meta(m_s['info']['id'], settings={'mode': m_s['mode']})
    model_meta = sly.ProjectMeta.from_json(model_meta_json)
    dst_meta = dst_meta.merge(model_meta)
    model_src_meta = dst_meta.to_json()

api.project.update_meta(dst_project_id, dst_meta.to_json())


def process(img, ann_json, model_settings):
    settings = {'annotation': ann_json, 'meta': model_settings['src_meta'], 'mode': model_settings['mode']}
    response = api.model.inference(model_settings['info']['id'], img, settings=settings)
    return img, response


for dataset_info in api.dataset.get_list(src_project_id):
    src_dataset_id = dataset_info['id']
    src_dataset_name = dataset_info['name']

    print('Project/Dataset: {}/{}'.format(src_project_name, src_dataset_name))

    dst_dataset_name = api.dataset.get_free_name(src_dataset_name, dst_project_id)
    dst_dataset_id = api.dataset.create(dst_dataset_name, dst_project_id)['id']

    for image_info in tqdm(api.image.get_list(src_dataset_id)):
        src_image_ext = image_info['meta']['mime'].split('/')[1]

        img = api.image.download_np(image_info['id'])
        ann_json = api.annotation.download(src_dataset_id, image_info['id'])

        for model in models_settings:
            img, ann_json = process(img, ann_json, model)

        inf_image, inf_ann_json = img, ann_json
        dst_img_name = api.image.get_free_name(image_info['name'], dst_dataset_id)
        dst_img_hash = api.image.upload_np(src_image_ext, inf_image)['hash']
        dst_img_id = api.image.add(dst_img_name, dst_img_hash, dst_dataset_id)['id']
        api.annotation.upload(inf_ann_json, dst_img_id)
