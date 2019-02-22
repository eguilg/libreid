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
from collections import defaultdict

from .bases import BaseImageDataset


class ZYDQ(BaseImageDataset):
	"""
	ZYDQ


	Dataset statistics:
	# identities: 1501 (+1 for background)
	# images: 12936 (train) + 3368 (query) + 15913 (gallery)
	"""
	dataset_dir = 'zydq'

	def __init__(self, root='data', verbose=True, **kwargs):
		super(ZYDQ, self).__init__()
		self.dataset_dir = osp.join(root, self.dataset_dir)

		self.total_dirs = [osp.join(self.dataset_dir, 'detections_half'),
						   osp.join(self.dataset_dir, 'detections_half2')]

		self._check_before_run()

		train = []
		query, gallery = self._process_dir(self.total_dirs, relabel=False)

		if verbose:
			print("=> zydq loaded")
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
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		for dir in self.total_dirs:
			if not osp.exists(dir):
				raise RuntimeError("'{}' is not available".format(dir))

	def _process_dir(self, dir_path, relabel=False):
		img_paths = []
		for dir in dir_path:
			img_paths += glob.glob(osp.join(dir, '*.png'))
		pattern = re.compile(r'^([\s\S]+)_([0-9]+)_([0-9]+).png$')

		cid_container = defaultdict(int)
		for img_path in img_paths:
			img_name = osp.basename(img_path)
			cname, _, _ = pattern.search(img_name).groups()
			cid_container[cname] += 1
		cnames, counts = zip(*cid_container.items())
		# query_cid = counts.index(min(counts))
		query_cnames = [
			'2号楼电梯口_2号一楼大厅_20190107165934_20190107171123_1218783',
			'2号楼电梯口_2号一楼大厅_20190107121913_20190107124511_1198772'
		]
		# query_cid = cnames.index('2号楼电梯口_2号一楼大厅_20190107165934_20190107171123_1218783')
		# query_cid = cnames.index('2号楼电梯口_2号一楼大厅_20190107121913_20190107124511_1198772')
		query, gallery = [], []
		for img_path in img_paths:
			img_name = osp.basename(img_path)
			cname, _, _ = pattern.search(img_name).groups()
			if cname in query_cnames:
				query.append((img_path, 0, cnames.index(cname)))
			else:
				gallery.append((img_path, 0, cnames.index(cname)))

		return query, gallery


class ZYDQ_Labeled(BaseImageDataset):
	"""
	ZYDQ_Labeled


	Dataset statistics:
	# identities: 20
	# images: 251 (train) + 0 (query) + 0 (gallery)
	"""
	dataset_dir = 'zydq'

	def __init__(self, root='data', verbose=True, upsample=64, **kwargs):
		super(ZYDQ_Labeled, self).__init__()
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.upsample = upsample
		self.train_dir = osp.join(self.dataset_dir, 'train')
		# self.query_dir = osp.join(self.dataset_dir, 'query')
		# self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
		self._check_before_run()

		train = self._process_dir(self.train_dir, relabel=True) * self.upsample
		query = []
		gallery = []
		if verbose:
			print("=> zydq_labeled loaded")
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
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))

		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))

	def _process_dir(self, dir_path, relabel=False):

		img_paths = glob.glob(osp.join(dir_path, '*.png'))
		pattern = re.compile(r'^([\s\S]+)_([0-9]+)_([0-9]+)_([0-9]+).png$')

		cid_container = defaultdict(int)
		pid_container = set()
		for img_path in img_paths:
			img_name = osp.basename(img_path)
			cname, _, _, pid = pattern.search(img_name).groups()
			pid = int(pid)
			cid_container[cname] += 1
			if pid == -1: continue  # junk images are just ignored
			pid_container.add(pid)
		cnames, counts = zip(*cid_container.items())
		pid2label = {pid: label for label, pid in enumerate(pid_container)}
		print(pid2label)

		dataset = []
		for img_path in img_paths:
			img_name = osp.basename(img_path)
			cname, _, _, pid = pattern.search(img_name).groups()
			pid = int(pid)
			if pid == -1: continue  # junk images are just ignored
			if relabel: pid = pid2label[pid]
			dataset.append((img_path, pid, cnames.index(cname)))

		return dataset
