import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
	def __init__(self, file_path, time_interval=0, transform=None):

		try:
			self.cv_cap = cv2.VideoCapture(file_path)
		except:
			raise FileNotFoundError('Error open video file %s'.format(file_path))
		if not self.cv_cap.isOpened():
			raise FileNotFoundError('Error open video file %s'.format(file_path))

		self.file_path = file_path
		self._transform = transform

		self.frame_count = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.h = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.w = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.fps = int(self.cv_cap.get(cv2.CAP_PROP_FPS))

		#  set time interval
		self.time_interval = time_interval
		self._frame_interval = 1
		if self.time_interval != 0:
			self._frame_interval = int(self.fps * self.time_interval)

		self._curr_frame_count = 0

		print('-' * 80)
		print('Video File: %s\n'
			  'Frame Count: %d\n'
			  'FPS: %d\n'
			  'H/W: %d/%d' % (file_path, self.frame_count,
							  self.fps, self.h, self.w))
		print('-' * 80)

	def __getitem__(self, item):
		out_frame_org = None
		out_frame = None
		time_stamp = self._curr_frame_count / self.fps
		for i in range(self._frame_interval):
			# read one frame from video
			ret, frame = self.cv_cap.read()
			if ret:  # if got a frame
				if i == 0:
					time_stamp = self._curr_frame_count / self.fps
					frame = frame[:, :, -1::-1].copy()  # (h, w, (bgr)) -> (h, w, (rgb))
					out_frame_org = frame
				self._curr_frame_count += 1
			else:
				break
		if self._transform is not None:
			out_frame = self._transform(out_frame_org)
			return time_stamp, out_frame_org, out_frame
		else:
			return time_stamp, out_frame_org

	def __len__(self):
		return (self.frame_count - 10) // self._frame_interval
