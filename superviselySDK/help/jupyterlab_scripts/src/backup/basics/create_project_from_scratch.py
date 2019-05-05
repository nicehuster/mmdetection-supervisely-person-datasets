# coding: utf-8
import supervisely_lib as sly
import os

address = os.environ['SERVER_ADDRESS']
token = os.environ['API_TOKEN']

print("Server address: ", address)
print("Your API token: ", token)

api = sly.Api(address, token)


team = api.team.get_list()[0]
workspace = api.workspace.get_list(team.id)[0]
print("Current team: id = {}, name = {!r}".format(team.id, team.name))
print("Current workspace: id = {}, name = {!r}".format(workspace.id, workspace.name))


project_name = 'project_test'
if api.project.exists(workspace.id, project_name):
    project_name = api.project.get_free_name(workspace.id, project_name)
project = api.project.create(workspace.id, project_name, 'project is created using IPython and Supervisely API')
print('Project {!r} has been sucessfully created: '.format(project.name))
print(project)


images = sly.fs.list_files('/workdir/demo_data/images')
# let's put first two images to the first dataset "ds1", and the rest of the images - to the second dataset "ds2

datasets = ['ds1', 'ds2']
dataset_images = [images[0:2], images[2:]]

for ds_name, img_paths in zip(datasets, dataset_images):
    ds = api.dataset.create(project.id, ds_name)
    print('Dataset {!r} has been sucessfully creates: id={}'.format(ds.name, ds.id))
    for img_path in img_paths:
        img_hash = api.image.upload_path(img_path)
        image_info = api.image.add(ds.id, sly.fs.get_file_name(img_path), img_hash)
        print('Image (id={}, name={}) has been sucessfully added'.format(image_info.id, image_info.name))

print("Number of images in created projects: ", api.project.get_images_count(project.id))

#define object classes
class_person = sly.ObjClass('person', sly.Rectangle, color=[255, 0, 0])
class_car = sly.ObjClass('car', sly.Polygon, color=[0, 255, 0])
class_road = sly.ObjClass('road', sly.Bitmap, color=[0, 0, 255])
obj_class_collection = sly.ObjClassCollection([class_person, class_car, class_road])

#define tags for images
tagmeta_weather = sly.TagMeta(name='weather',
                              value_type=sly.TagValueType.ONEOF_STRING,
                              possible_values=['rain', 'sun', 'cloud'],
                              color=[153, 0, 153])
tagmeta_annotate = sly.TagMeta('to_annotation', sly.TagValueType.NONE)
img_tag_meta_collection = sly.TagMetaCollection([tagmeta_weather, tagmeta_annotate])

#define tags for objects
tagmeta_vehicle_type = sly.TagMeta('vehicle_type', sly.TagValueType.ONEOF_STRING, ['sedan', 'suv', 'hatchback'])
tagmeta_confidence = sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER)
obj_tag_meta_collection = sly.TagMetaCollection([tagmeta_vehicle_type, tagmeta_confidence])

#combine everythiong to project meta
meta = sly.ProjectMeta(obj_class_collection, img_tag_meta_collection, obj_tag_meta_collection)
print(meta)
api.project.update_meta(project.id, meta.to_json())




