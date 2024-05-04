from sensor_msgs.msg import CameraInfo

class SpotCameraInfo(CameraInfo):
	def __init__(self):
		super().__init__()
		self.height = 360
		self.width = 640
		self.distortion_model = "plumb_bob"
		self.D = [0.0, 0.0, 0.0, 0.0, 0.0]
		self.K = [266.84222412109375, 0.0, 322.45416259765625, 0.0, 266.84222412109375, 186.8440399169922, 0.0, 0.0, 1.0]
		self.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		self.P = [266.84222412109375, 0.0, 322.45416259765625, 0.0, 0.0, 266.84222412109375, 186.8440399169922, 0.0, 0.0, 0.0, 1.0, 0.0]
		self.binning_x = 0 
		self.binning_y = 0

class WartyCameraInfo(CameraInfo):
	def __init__(self):
		super().__init__()
		self.D = [0.0, 0.0, 0.0, 0.0, 0.0]
		self.K = [260.99805320956386, 0.0, 320.0, 0.0, 260.99805320956386, 240.0, 0.0, 0.0, 1.0]
		self.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		self.P = [260.99805320956386, 0.0, 320.0, 0.0, 0.0, 260.99805320956386, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

class HuskyCameraInfo(CameraInfo):
	def __init__(self):
		super().__init__()
		self.height = 240
		self.width = 320
		self.distortion_model = "plumb_bob"
		self.D = [0.0, 0.0, 0.0, 0.0, 0.0]
		self.K = [386.27425923951824, 0.0, 160.5, 0.0, 386.27425923951824, 120.5, 0.0, 0.0, 1.0]
		self.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		self.P = [386.27425923951824, 0.0, 160.5, -27.039198146766278, 0.0, 386.27425923951824, 120.5, 0.0, 0.0, 0.0, 1.0, 0.0]
		self.binning_x = 0 
		self.binning_y = 0