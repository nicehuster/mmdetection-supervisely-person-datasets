# coding: utf-8
address = 'http://192.168.1.69:5555'
token = 'OfaV5z24gEQ7ikv2DiVdYu1CXZhMavU7POtJw2iDtQtvGUux31DUyWTXW6mZ0wd3IRuXTNtMFS9pCggewQWRcqSTUi4EJXzly8kH7MJL1hm3uZeM2MCn5HaoEYwXejKT'
team_name = 'max'
workspace_name = 'script1'

src_project_name = 'lemons_annotated'
dst_project_name = 'lemons_train_detection'

validation_portion = 0.05
multiplier = 5

import supervisely_lib as sly
from tqdm import tqdm
import random

api = sly.Api(address, token)

team_id = api.team.get_id_by_name(team_name)
workspace_id = api.workspace.get_id_by_name(workspace_name, team_id)


src_project_id = api.project.get_id_by_name(src_project_name, workspace_id)
src_meta_json = api.project.get_meta(src_project_id)
src_meta = sly.ProjectMeta.from_json(src_meta_json)


def process_meta(input_meta):
    classes_mapping = {}
    output_meta = sly.ProjectMeta(obj_classes=[], img_tag_metas=input_meta.img_tag_metas, obj_tag_metas=input_meta.obj_tags)
    for obj_class in input_meta.obj_classes:
        classes_mapping[obj_class.name] = '{}_bbox'.format(obj_class.name)
        new_obj_class = sly.ObjClass(classes_mapping[obj_class.name], sly.Rectangle, color=obj_class.color)
        output_meta = output_meta.add_obj_class(new_obj_class)
    output_meta = output_meta.add_img_tag_meta(sly.TagMeta('train', sly.TagValueType.NONE))
    output_meta = output_meta.add_img_tag_meta(sly.TagMeta('val', sly.TagValueType.NONE))
    return output_meta, classes_mapping


dst_project_name = api.project.get_free_name(dst_project_name, workspace_id)
dst_project_id = api.project.create(dst_project_name, workspace_id)[api.project.ID]
dst_meta, classes_mapping = process_meta(src_meta)
api.project.update_meta(dst_project_id, dst_meta.to_json())


def process(img, ann):
    results = []

    ann_new = ann.clone()

    for label in ann.labels:
        new_class = dst_meta.get_obj_class(classes_mapping[label.obj_class.name])
        new_geometry = label.geometry.to_bbox()
        new_label = label.clone(obj_class=new_class, geometry=new_geometry)
        ann_new.add_label(new_label)

    results.append((img, ann_new))

    img_lr, ann_lr = sly.aug.fliplr(img, ann_new)
    results.append((img_lr, ann_lr))

    crops = []
    for cur_img, cur_ann in results:
        for i in range(multiplier):
            res_img, res_ann = sly.aug.random_crop_fraction(cur_img, cur_ann, (70, 90), (70, 90))
            crops.append((res_img, res_ann))
    results.extend(crops)

    for cur_img, cur_ann in results:
        tag = sly.Tag(dst_meta.get_img_tag_meta('train'), value=None)
        if random.random() <= validation_portion:
            tag = sly.Tag(dst_meta.get_img_tag_meta('val'), value=None)
        cur_ann.add_tag(tag)
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
        ann_json = api.annotation.download(src_dataset_id, image_info['id'])['annotation']
        ann = sly.Annotation.from_json(ann_json, src_meta)

        aug_results = process(img, ann)

        for aug_img, aug_ann in aug_results:
            dst_img_name = api.image.get_free_name(image_info['title'], dst_dataset_id)
            dst_img_hash = api.image.upload_np(src_image_ext, aug_img)['hash']
            dst_img_id = api.image.add(dst_img_name, dst_img_hash, dst_dataset_id)['id']
            api.annotation.upload(aug_ann.to_json(), dst_img_id)

    print()
