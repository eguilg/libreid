import argparse
import os
import os.path as osp
from collections import defaultdict
from darknet.frameloader import VideoDataset
from darknet.detector import Detector
from darknet.preprocess import prep_image
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


def arg_parse():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	#  data dirs
	parser.add_argument('-s', '--source-dir', type=str, required=True, nargs='+',
						help="source dir or path (delimited by space)")
	parser.add_argument('--file-filters', default=['.mp4'], type=str, nargs='+',
						help="source dir or path (delimited by space)")
	parser.add_argument('-t', '--target-dir', type=str, required=True,
						help="target bbox save dir")

	#  model
	parser.add_argument('--cfg-path', type=str, required=True,
						help="path to model config file")

	parser.add_argument('--weight-path', type=str, required=True,
						help="path to model weight file")

	parser.add_argument('--names-path', type=str, required=True,
						help="path to class names file")

	#  others
	parser.add_argument('--batch-size', default=10, type=int,
						help="batch size")
	parser.add_argument('--reso', default=416, type=int,
						help="input resolution")
	parser.add_argument('--time-interval', default=4, type=int,
						help="detect time interval")
	parser.add_argument('--conf', default=0.7, type=float,
						help="confidence")
	parser.add_argument('--nms_conf', default=0.4, type=float,
						help="nms confidence")

	parser.add_argument('--half', action='store_true',
						help="half the model data type")
	parser.add_argument('--use-cpu', action='store_true',
						help="use cpu")
	parser.add_argument('--gpu-devices', default='0', type=str,
						help='gpu device ids for CUDA_VISIBLE_DEVICES')

	return parser.parse_args()


def listdir_r(dir):
	paths = []
	if osp.isfile(dir):
		paths.append(dir)
	elif osp.isdir(dir):
		for subdir in os.listdir(dir):
			paths.extend(listdir_r(osp.join(dir, subdir)))

	return paths


def filter_paths(paths, suffixs):
	filtered_paths = set()
	for suffix in suffixs:
		filtered_paths.update(list(filter(lambda path: path.endswith(suffix), paths)))
	return list(filtered_paths)


def check_dirs(args):
	target_dir = args.target_dir
	if not osp.isdir(target_dir):
		raise NotADirectoryError('target-path %s is NOT a Directory' % target_dir)

	source_dirs = args.source_dir
	source_paths = []
	for dir in source_dirs:
		source_paths.extend(listdir_r(dir))
	source_paths = filter_paths(source_paths, args.file_filters)

	if len(source_paths) == 0:
		raise FileNotFoundError('source dirs are empty:', source_dirs)
	else:
		print('Videos to detect: ')
		for path in source_paths:
			print(path)

	return source_paths, target_dir


def init(args):
	source_paths, target_dir = check_dirs(args)

	detector = Detector(args.cfg_path, args.weight_path, args.names_path, args.reso)

	data_loaders = [DataLoader(VideoDataset(file_path=vpath,
											time_interval=args.time_interval,
											transform=detector.transforms),
							   batch_size=args.batch_size)
					for vpath in source_paths]

	return detector, data_loaders, target_dir


def detect_batch(input_batch, origin_batch, detector, target_names, CUDA, confidence, nms_conf):
	#  get origin w h
	im_dim = torch.FloatTensor([origin_batch.size(2), origin_batch.size(1)])
	if detector.model.is_half:
		im_dim = im_dim.half()
	if CUDA:
		im_dim = im_dim.cuda()
	output = detector.detect_batch(input_batch, target_names, CUDA, confidence, nms_conf)
	output = detector.get_batch_bboxes(im_dim, output)  # idx, x1, y1, x2, y2, ...

	if CUDA:
		output = output.detach().cpu()

	output = output.numpy()
	origin_batch = origin_batch.numpy()
	idxs = output[:, 0].astype(int).tolist()
	bboxes = output[:, 1:5].astype(int).tolist()

	out_imgs = []
	out_idxs = []
	ToPIL = transforms.ToPILImage()
	for idx, bbox in zip(idxs, bboxes):
		w = bbox[2] - bbox[0]
		h = bbox[3] - bbox[1]
		if w > 0 and h > 0 and h / w >= 1.2:
			org_img = ToPIL(origin_batch[idx])
			crop_img = org_img.crop(bbox)

			out_idxs.append(idx)
			out_imgs.append(crop_img)

	return out_idxs, out_imgs


def main():
	args = arg_parse()
	CUDA = not args.use_cpu
	HALF = args.half
	target_names = ['person']
	detector, data_loaders, target_dir = init(args)

	if HALF:
		detector.half()
	if CUDA:
		detector.cuda()

	for loader in data_loaders:
		vfile_path = loader.dataset.file_path
		print('-' * 80)
		print('Detecting file: %s' % vfile_path)
		vfile_name = ''.join(osp.basename(vfile_path).split('.')[:-1])
		timestamp_records = defaultdict(int)
		for timestamp, org_batch, input_batch in loader:
			input_batch = input_batch.float()
			if HALF:
				input_batch = input_batch.half()
			if CUDA:
				input_batch = input_batch.cuda()
			timestamp = timestamp.numpy().astype(int).tolist()
			idxs, out_imgs = detect_batch(input_batch, org_batch, detector, target_names, CUDA, args.conf,
										  args.nms_conf)
			for idx, out_imgs in zip(idxs, out_imgs):
				detection_name = '_'.join([vfile_name, str(timestamp[idx]), str(timestamp_records[timestamp[idx]])])
				detection_path = osp.join(target_dir, detection_name + '.png')
				out_imgs.save(detection_path)
				print('Saved detection at %d s to %s' % (timestamp[idx], detection_path))
				timestamp_records[timestamp[idx]] += 1

		n_detection = sum(timestamp_records.values())
		print('Detected %d instances in %d detections' % (n_detection, len(timestamp_records)))
		print('-' * 80)
	print('Done')


if __name__ == '__main__':
	main()
