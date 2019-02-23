from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset

# To adapt to different versions
# Log:
# 22.01.2019: v1 and v2 only differ in dir names
_TRAIN_DIR_KEY = 'train_dir'
_TEST_DIR_KEY = 'test_dir'
_VERSION = {
	'MSMT17_V1': {
		_TRAIN_DIR_KEY: 'train',
		_TEST_DIR_KEY: 'test',
	},
	'MSMT17_V2': {
		_TRAIN_DIR_KEY: 'mask_train_v2',
		_TEST_DIR_KEY: 'mask_test_v2',
	}
}


class MSMT17(BaseImageDataset):
	"""
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
	dataset_dir = 'msmt17'

	def __init__(self, root='data', verbose=True, **kwargs):
		super(MSMT17, self).__init__()
		self.dataset_dir = osp.join(root, self.dataset_dir)
		has_main_dir = False
		for main_dir in _VERSION:
			if osp.exists(osp.join(self.dataset_dir, main_dir)):
				train_dir = _VERSION[main_dir][_TRAIN_DIR_KEY]
				test_dir = _VERSION[main_dir][_TEST_DIR_KEY]
				has_main_dir = True
				break
		assert has_main_dir, 'Dataset folder not found'
		self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
		self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
		self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
		self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
		self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
		self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')

		self._check_before_run()
		train = self._process_dir(self.train_dir, self.list_train_path)
		# val = self._process_dir(self.train_dir, self.list_val_path)
		query = self._process_dir(self.test_dir, self.list_query_path)
		gallery = self._process_dir(self.test_dir, self.list_gallery_path)

		# To fairly compare with published methods, don't use val images for training
		# train += val
		# num_train_imgs += num_val_imgs

		if verbose:
			print('=> MSMT17 loaded')
			self.print_dataset_statistics(train, query, gallery)

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
		self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
		self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError('"{}" is not available'.format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError('"{}" is not available'.format(self.test_dir))

	def _process_dir(self, dir_path, list_path):
		with open(list_path, 'r') as txt:
			lines = txt.readlines()
		dataset = []
		pid_container = set()
		for img_idx, img_info in enumerate(lines):
			img_path, pid = img_info.split(' ')
			pid = int(pid)  # no need to relabel
			camid = int(img_path.split('_')[2]) - 1  # index starts from 0
			img_path = osp.join(dir_path, img_path)
			dataset.append((img_path, pid, camid))
			pid_container.add(pid)
		num_pids = len(pid_container)
		# check if pid starts from 0 and increments with 1
		for idx, pid in enumerate(pid_container):
			assert idx == pid, 'See code comment for explanation'
		return dataset