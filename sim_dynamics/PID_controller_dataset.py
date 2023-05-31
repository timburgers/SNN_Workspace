import matplotlib.pyplot as plt
import numpy as np
from Dynamics import Blimp
import math
import random
import yaml

#GLOBAL PARAMS
TIME_STEP = 0.01	# The sample time per time step [s]
SIM_TIME = 100		# Total length of simulation [s]
SETPOINT_UPDATE_STEP = 5
MIMIMAL_HEIGHT_CHANGE = 5
RANDOM_SEED = 2


SETPOINT_Z = 0 		# Setpoint of height [m]
Z_INITIAL = 0 		# Initial height [m]
DZ_INITIAL = 0 		# Initial velocity [m/s]

MASS = 1 			# Total mass of drone [kg]
MAX_THRUST = 1500 	# Max allowed total thrust of drone [N]

antiWindup = False	# If set to true, no windup will take place when the trust limit is reached in the integration error


#---PID GAINS--- 
KP = 3
KI = 0
KD = 3
#---------------

random.seed(RANDOM_SEED)




class Simulation_PID(object):
	def __init__(self):
		global SETPOINT_Z
		SETPOINT_Z = 0 	
		with open("configs/config_LIF_DEFAULT.yaml","r") as f:
			self.config = yaml.safe_load(f)
		self.states = Blimp(self.config)
		
		
		self.sim = True
		self.timer = 0
		self.prev_interval = -1
		
		
		self.z = np.array([])
		self.dz = np.array([])
		self.velocity = np.array([])
		self.times = np.array([])
		self.kpe = np.array([])
		self.kde = np.array([])
		self.kie = np.array([])
		self.thrst = np.array([])
		self.z_ref = np.array([])
		self.error = np.array([])
	
	def PID_cycle(self):
		self.pid = PID(KP, KI, KD)

		
		while(self.sim):
			
			thrust = self.pid.compute(SETPOINT_Z, self.states.get_z())

			if self.timer > SIM_TIME/TIME_STEP:
				print("SIM ENDED")
				self.sim = False

			self.times = np.append(self.times,self.timer)
			self.z = np.append(self.z,self.states.get_z())
			self.dz = np.append(self.dz,self.states.get_dz())
			self.kpe = np.append(self.kpe,self.pid.get_kpe())
			self.kde = np.append(self.kde,self.pid.get_kde())
			self.kie = np.append(self.kie,self.pid.get_kie())
			self.thrst = np.append(self.thrst,thrust)
			self.z_ref = np.append(self.z_ref,SETPOINT_Z)
			self.error = np.append(self.error,SETPOINT_Z-self.states.get_z())

			# Calculate the effect of the thrust on the height
			self.states.sim_dynamics(thrust)

			self.timer += 1

			# Calculate updated setpoint 
			self.prev_interval = update_setpoint(SETPOINT_UPDATE_STEP, self.timer, TIME_STEP, self.prev_interval)


		# graph(self.times,self.z,self.kpe,self.kde,self.kie,self.thrst,self.z_ref)
		save_data(self.z,self.z_ref,self.error,self.kpe,self.kde,self.thrst)

def update_setpoint(freq_update, timer, dt, prev_interval):
	global SETPOINT_Z
	new_setpoint = 0
	current_interval = math.floor(timer*dt/freq_update)
	
	if current_interval != prev_interval:
		while (new_setpoint == 0 or abs(new_setpoint-SETPOINT_Z) <= MIMIMAL_HEIGHT_CHANGE/10):
			new_setpoint = random.uniform(0,1)
		SETPOINT_Z = new_setpoint
	prev_interval = current_interval

	return prev_interval

def graph(x,z1,z2,z3,z4,z5,z_ref):
	x = x*TIME_STEP
	fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True)
	ax1.set(ylabel='Drone \nHeight')
	ax1.grid()
	ax1.plot(x,z1)
	ax1.plot(x,z_ref, linestyle = '-')
	ax2.set(ylabel='KP_error')
	ax2.plot(x,z2,'tab:red')
	ax2.grid()
	ax3.set(ylabel='KD_error')
	z3[0] = z3[1]
	ax3.plot(x,z3,'tab:orange')
	ax3.grid()
	ax4.set(ylabel='KI_error')
	ax4.plot(x,z4,'tab:pink')
	ax4.grid()
	ax5.set(ylabel='Drone \nThrust')
	ax5.plot(x,z5,'tab:brown')
	ax5.grid()
	plt.show()

def save_data(z,z_ref,error,kpe,kde, T):
	z.shape = [len(z),1]
	z_ref.shape= [len(z_ref),1]
	error.shape=[len(error),1]
	kpe.shape = [len(kpe),1]
	kde.shape = [len(kde),1]
	T.shape = [len(T),1]
	
	np.savetxt("Sim_data/height_control_PID/z_zref_error_kpe_kde_Thrust/dataset_" + str(idx)+ ".csv",np.concatenate([z,z_ref,error,kpe,kde,T],axis=1) , delimiter=',', header= "timestep = " + str(TIME_STEP)+ ", sim time = "+ str(SIM_TIME)+ ", new_ref_freq = "+ str(SETPOINT_UPDATE_STEP)+ ", minimal_height_change = " + str(MIMIMAL_HEIGHT_CHANGE))

class PID(object):
	def __init__(self,KP,KI,KD):
		self.kp = KP
		self.ki = KI
		self.kd = KD
		
		self.error = 0
		self.integral_error = 0
		self.error_last = 0
		self.derivative_error = 0
		self.output = 0
		# self.setpoint_update = True
		self.prev_setpoint = None
		
	def compute(self, setpoint, measure):
		self.error = setpoint - measure

		if setpoint != self.prev_setpoint:
			self.setpoint_update = True
		self.prev_setpoint = setpoint

		if self.setpoint_update == True:
			self.derivative_error = 0
		else:
			self.derivative_error = (self.error - self.error_last)/TIME_STEP
		self.setpoint_update = False
			
		
		self.error_last = self.error
		

		if(antiWindup and abs(self.output)>= MAX_THRUST and (((self.error>=0) and (self.integral_error>=0))or((self.error<0) and (self.integral_error<0)))):
			#no integration
			self.integral_error = self.integral_error

		else:
			#rectangular integration
			self.integral_error = self.integral_error + self.error * TIME_STEP

		
		# Control output + hover thrust
		self.output = self.kp * self.error + self.ki*self.integral_error + self.kd*self.derivative_error
		#print(self.output)

		#Saturation limits motor
		if self.output >= MAX_THRUST:
			self.output = MAX_THRUST
		# elif self.output <= 0:
		# 	self.output = 0
		return self.output
		
	def get_kpe(self):
		return self.kp * self.error
	def get_kde(self):
		return self.kd * self.derivative_error
	def get_kie(self):
		return self.ki * self.integral_error

def main():
	sim = Simulation_PID()
	sim.PID_cycle()

idx = 0
for SETPOINT_UPDATE_STEP in [3,4,5,6]:
	for MIMIMAL_HEIGHT_CHANGE in [1,2,3,4]:
		for RANDOM_SEED in range(32):
			main()
			idx += 1

# main()