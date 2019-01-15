import os
from .util import *
from .darknet import Darknet
from .preprocess import prep_image, inp_to_image
from torchvision import transforms
from PIL import Image


class Detector(object):
	def __init__(self, cfg_path, weight_path, reso=416):
		# Set up the neural network
		print("Loading network config.....")
		self.model = Darknet(cfg_path)

		if os.path.isfile(weight_path):
			print("Loading network weight.....")
			self.model.load_weights(weight_path)
			print("Network successfully loaded")

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

	def detect_batch(self, batch_img, target_class_idxs, CUDA, confidence=0.4, nms_conf=0.4):
		assert batch_img.size(2) == self.input_size
		assert batch_img.size(3) == self.input_size

		self.model.eval()
		output = self.model(batch_img, CUDA)  # bs, a*a*15(a*a area a=in_size//32, 15 boxes each), 5 (xywh c)
		output = write_results(output, confidence, target_class_idxs, 0 < nms_conf < 1, nms_conf)

		return output
