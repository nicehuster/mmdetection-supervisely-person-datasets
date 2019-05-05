# coding: utf-8
address = 'http://192.168.1.69:5555'
token = 'YGPDnuBkhFmcQ7VNzSEjhgavjg4eFR4Eq1C3jIY4HgV3SQq2JgkXCNtgZy1Fu2ftd4IKui8DsjrdtXjB853cMtBevpSJqFDYiaG1A5qphlH6fFiYYmcVZ5fMR8dDrt5l'
team_name = 'dima'
workspace_name = 'work'

src_project_name = 'lemons_df'
dst_project_name = 'lemons_out'

validation_portion = 0.05
multiplier = 5


import supervisely_lib as sly
from tqdm import tqdm


from data_verification import verify_data, make_false_negative_name, make_false_positive_name, \
    make_iou_tag_name

api = sly.Api(address, token)

team_id = api.team.get_info_by_name(team_name)['id']
workspace_id = api.workspace.get_info_by_name(workspace_name, team_id)['id']

dst_project_name = api.project.get_free_name(dst_project_name, workspace_id)
dst_project_id = api.project.create(dst_project_name, workspace_id)['id']

src_project_id = api.project.get_info_by_name(src_project_name, workspace_id)['id']

src_meta_json = api.project.get_meta(src_project_id)
src_meta = sly.ProjectMeta.from_json(src_meta_json)

classes_mapping = {
    'lemon': 'lemon_pred',
    'kiwi': 'kiwi_pred'
}


def process_meta(input_meta):
    output_meta = sly.ProjectMeta(obj_classes=None, img_tag_metas=input_meta.img_tag_metas, objtag_metas=input_meta.obj_tags)
    for obj_class in input_meta.obj_classes:
        if obj_class.name in classes_mapping.keys() or obj_class.name in classes_mapping.values():
            output_meta = output_meta.add_obj_class(obj_class)

    for gt_class in classes_mapping:
        output_meta = output_meta.add_obj_class(sly.ObjClass(make_false_positive_name(gt_class), sly.Bitmap))
        output_meta = output_meta.add_obj_class(sly.ObjClass(make_false_negative_name(gt_class), sly.Bitmap))
        output_meta = output_meta.add_img_tag_meta(sly.TagMeta(make_iou_tag_name(gt_class), sly.TagValueType.ANY_NUMBER))
    return output_meta


dst_meta = process_meta(src_meta)
api.project.update_meta(dst_project_id, dst_meta.to_json())


def process(img, ann):
    results = []
    ann = verify_data(ann, classes_mapping, dst_meta)
    results.append((img, ann))
    return results


for dataset_info in api.dataset.get_list(src_project_id):
    src_dataset_id = dataset_info['id']
    src_dataset_name = dataset_info['title']

    print('Project/Dataset: {}/{}'.format(src_project_name, src_dataset_name))

    dst_dataset_name = api.dataset.get_free_name(src_dataset_name, dst_project_id)
    dst_dataset_id = api.dataset.create(dst_dataset_name, dst_project_id)['id']

    for image_info in tqdm(api.image.get_list(src_dataset_id)):
        src_image_ext = image_info['meta']['mime'].split('/')[1]

        img = api.image.download_np(image_info['id'])
        ann_json = api.annotation.download(src_dataset_id, image_info['id'])
        ann = sly.Annotation.from_json(ann_json, src_meta)

        aug_results = process(img, ann)

        for aug_img, aug_ann in aug_results:
            dst_img_name = api.image.get_free_name(image_info['title'], dst_dataset_id)
            dst_img_hash = api.image.upload_np(src_image_ext, aug_img)['hash']
            dst_img_id = api.image.add(dst_img_name, dst_img_hash, dst_dataset_id)['id']
            api.annotation.upload(aug_ann.to_json(), dst_img_id)

    print()
