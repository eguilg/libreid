import os
from .util import *
from .darknet import Darknet
from .preprocess import prep_image, inp_to_image
from torchvision import transforms
from PIL import Image


class Detector(object):
	def __init__(self, cfg_path, weight_path, names_path, reso=416):
		# Set up the neural network
		if os.path.isfile(weight_path):
			print("Loading network config.....")
			self.model = Darknet(cfg_path)
		else:
			raise FileNotFoundError('model config file %s not found' % cfg_path)

		if os.path.isfile(weight_path):
			print("Loading network weight.....")
			self.model.load_weights(weight_path)
			print("Network successfully loaded")
		else:
			raise FileNotFoundError('model weight file %s not found' % weight_path)

		if os.path.isfile(names_path):
			print("Loading classes names......")
			self.class_names = load_classes(names_path)
		else:
			raise FileNotFoundError('class names file %s not found' % names_path)

		self.model.net_info["height"] = reso

		self.input_size = int(self.model.net_info["height"])
		self.num_classes = self.model.net_info
		assert self.input_size % 32 == 0
		assert self.input_size > 32
		self.model.eval()

		self.transforms = transforms.Compose([
			transforms.Lambda(lambda img: prep_image(img, self.input_size)[0]),
			transforms.ToTensor()
		])

	def cuda(self):
		self.model = self.model.cuda()

	def eval(self):
		self.model.eval()

	def detect_batch(self, batch_img, target_class_names, CUDA, confidence=0.4, nms_conf=0.4):
		assert batch_img.size(2) == self.input_size
		assert batch_img.size(3) == self.input_size

		target_class_idxs = list(filter(lambda idx: self.class_names[idx] in target_class_names,
										range(len(self.class_names))))

		self.model.eval()
		output = self.model(batch_img, CUDA)  # bs, a*a*15(a*a area a=in_size//32, 15 anchors), 5 (xywh c)
		output = write_results(output, confidence, target_class_idxs, 0 < nms_conf < 1, nms_conf)

		return output

	def get_batch_bboxes(self, im_dim, output):
		inp_dim = self.input_size
		im_dim = im_dim.repeat(output.size(0), 1)

		#  factor to scale to original size
		scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

		#  remove the offsets, so that can fit on origin img coordinates
		output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
		output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

		#  scale to original size
		output[:, 1:5] /= scaling_factor

		for i in range(output.shape[0]):
			output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
			output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

		return output
