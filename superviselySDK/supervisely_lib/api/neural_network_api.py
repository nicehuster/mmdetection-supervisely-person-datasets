# coding: utf-8

import os
from collections import namedtuple
import tarfile
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import numpy as np
import json

from supervisely_lib.api.module_api import ApiField, ModuleApi
from supervisely_lib._utils import rand_str, camel_to_snake
from supervisely_lib.io.fs import ensure_base_path, silent_remove
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.project.project_meta import ProjectMeta


class NeuralNetworkApi(ModuleApi):
    _info_sequence = [ApiField.ID,
                      ApiField.NAME,
                      ApiField.DESCRIPTION,
                      ApiField.CONFIG,
                      ApiField.HASH,
                      ApiField.ONLY_TRAIN,
                      ApiField.PLUGIN_ID,
                      ApiField.PLUGIN_VERSION,
                      ApiField.SIZE,
                      ApiField.WEIGHTS_LOCATION,
                      ApiField.README,
                      ApiField.TASK_ID,
                      ApiField.USER_ID,
                      ApiField.TEAM_ID,
                      ApiField.WORKSPACE_ID,
                      ApiField.CREATED_AT,
                      ApiField.UPDATED_AT]
    Info = namedtuple('ModelInfo', [camel_to_snake(name) for name in _info_sequence])

    def get_list(self, workspace_id, filters=None):
        return self.get_list_all_pages('models.list',  {ApiField.WORKSPACE_ID: workspace_id, ApiField.FILTER: filters or []})

    def get_info_by_id(self, id):
        return self._get_info_by_id(id, 'models.info')

    def download(self, id):
        response = self.api.post('models.download', {ApiField.ID: id}, stream=True)
        return response

    def download_to_tar(self, workspace_id, name, tar_path, progress_cb=None):
        model = self.get_info_by_name(workspace_id, name)
        response = self.download(model.id)
        ensure_base_path(tar_path)
        with open(tar_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                if progress_cb is not None:
                    read_mb = len(chunk) / 1024.0 / 1024.0
                    progress_cb(read_mb)

    def download_to_dir(self, workspace_id, name, directory, progress_cb=None):
        model_tar = os.path.join(directory, rand_str(10) + '.tar')
        self.download_to_tar(workspace_id, name, model_tar, progress_cb)
        model_dir = os.path.join(directory, name)
        with tarfile.open(model_tar) as archive:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, model_dir)
        silent_remove(model_tar)
        return model_dir

    def generate_hash(self, task_id):
        response = self.api.post('models.hash.create', {ApiField.TASK_ID: task_id})
        return response.json()

    def upload(self, hash, archive_path, progress_cb=None):
        encoder = MultipartEncoder({'hash': hash,
                                    'weights': (os.path.basename(archive_path), open(archive_path, 'rb'), 'application/x-tar') })
        def callback(monitor):
            read_mb = monitor.bytes_read / 1024.0 / 1024.0
            if progress_cb is not None:
                progress_cb(read_mb)
        monitor = MultipartEncoderMonitor(encoder, callback)
        self.api.post('models.upload', monitor)

    def inference_remote_image(self, id, image_hash, ann=None, meta=None, mode=None, ext=None):
        data = {
            "request_type": "inference",
            "meta": meta or ProjectMeta().to_json(),
            "annotation": ann or None,
            "mode": mode or {},
            "image_hash": image_hash
        }
        fake_img_data = sly_image.write_bytes(np.zeros([5, 5, 3]), '.jpg')
        encoder = MultipartEncoder({'id': str(id).encode('utf-8'),
                                    'data': json.dumps(data),
                                    'image': ("img", fake_img_data, "")})
        response = self.api.post('models.infer', MultipartEncoderMonitor(encoder))
        return response.json()

    def inference(self, id, img, ann=None, meta=None, mode=None, ext=None):
        data = {
            "request_type": "inference",
            "meta": meta or ProjectMeta().to_json(),
            "annotation": ann or None,
            "mode": mode or {},
        }
        img_data = sly_image.write_bytes(img, ext or '.jpg')
        encoder = MultipartEncoder({'id': str(id).encode('utf-8'),
                                    'data': json.dumps(data),
                                    'image': ("img", img_data, "")})

        response = self.api.post('models.infer', MultipartEncoderMonitor(encoder))
        return response.json()

    def get_output_meta(self, id, input_meta=None, inference_mode=None):
        data = {
            "request_type": "get_out_meta",
            "meta": input_meta or ProjectMeta().to_json(),
            "mode": inference_mode or {}
        }
        encoder = MultipartEncoder({'id': str(id).encode('utf-8'),
                                    'data': json.dumps(data)})
        response = self.api.post('models.infer', MultipartEncoderMonitor(encoder))
        response_json = response.json()
        if 'out_meta' in response_json:
            return response_json['out_meta']
        return response.json()

    def get_deploy_tasks(self, model_id):
        response = self.api.post('models.info.deployed', {'id': model_id})
        return [task[ApiField.ID] for task in response.json()]

    def _clone_api_method_name(self):
        return 'models.clone'
