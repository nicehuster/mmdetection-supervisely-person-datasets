# coding: utf-8
import supervisely_lib as sly
from tqdm import tqdm
import random
import os

team_name = 'max'
workspace_name = 'test_dtl_segmentation'

src_project_name = 'lemons_annotated'
dst_project_name = 'lemons_annotated_segmentation'

validation_portion = 0.05
image_multiplier = 5

class_bg = sly.ObjClass('bg', sly.Rectangle)
tag_meta_train = sly.TagMeta('train', sly.TagValueType.NONE)
tag_meta_val = sly.TagMeta('val', sly.TagValueType.NONE)


address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)


team = api.team.get_info_by_name(team_name)
workspace = api.workspace.get_info_by_name(team.id, workspace_name)
print("Current context: Team {!r}, Workspace {!r}".format(team.name, workspace.name))

src_project = api.project.get_info_by_name(workspace.id, src_project_name)
src_meta_json = api.project.get_meta(src_project.id)
src_meta = sly.ProjectMeta.from_json(src_meta_json)


def process_meta(input_meta):
    output_meta = input_meta.clone()
    output_meta = output_meta.add_obj_class(class_bg)
    output_meta = output_meta.add_img_tag_meta(tag_meta_train)
    output_meta = output_meta.add_img_tag_meta(tag_meta_val)
    return output_meta


dst_meta = process_meta(src_meta)
print(dst_meta)

if api.project.exists(workspace.id, dst_project_name):
    dst_project_name = api.project.get_free_name(workspace.id, dst_project_name)
dst_project = api.project.create(workspace.id, dst_project_name)
api.project.update_meta(dst_project.id, dst_meta.to_json())


def process(img, ann):
    original = (img, ann)
    flipped = sly.aug.fliplr(*original)

    crops = []
    for cur_img, cur_ann in [original, flipped]:
        for i in range(image_multiplier):
            res_img, res_ann = sly.aug.random_crop_fraction(cur_img, cur_ann, (70, 90), (70, 90))
            crops.append((res_img, res_ann))

    results = []
    for cur_img, cur_ann in [original, flipped, *crops]:
        bg_label = sly.Label(sly.Rectangle.from_array(cur_img), class_bg)
        cur_ann = cur_ann.add_label(bg_label)
        tag = sly.Tag(tag_meta_train if random.random() <= validation_portion else tag_meta_val)
        cur_ann = cur_ann.add_tag(tag)
        results.append((cur_img, cur_ann))

    return results

aug_results_debug = None

print("Project {!r}: training data preparation".format(src_project.name))
for src_dataset in api.dataset.get_list(src_project.id):
    print('Dataset: {}'.format(src_dataset.name))
    dst_dataset = api.dataset.create(dst_project.id, src_dataset.name)

    for image in tqdm(api.image.get_list(src_dataset.id)):
        img = api.image.download_np(image.id)
        ann_info = api.annotation.download(image.id)
        ann = sly.Annotation.from_json(ann_info.annotation, src_meta)

        aug_results = process(img, ann)

        if aug_results_debug is None:
            aug_results_debug = aug_results.copy()

        for aug_img, aug_ann in aug_results:
            dst_img_name = api.image.get_free_name(dst_dataset.id, image.name)
            dst_img_hash = api.image.upload_np(aug_img, image.ext)
            dst_image = api.image.add(dst_dataset.id, dst_img_name, dst_img_hash)
            api.annotation.upload(dst_image.id, aug_ann.to_json())

print("Visualization of augmentation results for first image: ")
for aug_img, aug_ann in aug_results_debug:
    aug_ann.draw(aug_img)
