
class Drone(object):
	def __init__(self, Z_INITIAL, DZ_INITIAL, MASS, TIME_STEP,g):
		global Drone
		self.ddz = 0
		self.dz = DZ_INITIAL
		self.z = Z_INITIAL
		self.mass = MASS
		self.time_step = TIME_STEP
		self.g = g

	def sim_dynamics(self,thrust):
		self.ddz = self.g + thrust / self.mass
		self.dz += self.ddz * self.time_step
		self.z += self.dz * self.time_step

	def get_ddz(self):
		return self.ddz
	def get_dz(self):
		return self.dz
	def get_z(self):
		return self.z

class Blimp():
	def __init__(self,config,bias):
		self.bias = bias
		self.ddz = 0
		self.dz = config["DZ_INITIAL"]
		self.z = config["Z_INITIAL"]
		self.time_step = config["TIME_STEP"]

	def sim_dynamics(self,input):
		self.ddz = input - self.bias
		self.dz += self.ddz * self.time_step
		self.z += self.dz * self.time_step
		

	def get_ddz(self):
		return self.ddz
	def get_dz(self):
		return self.dz
	def get_z(self):
		return self.z 