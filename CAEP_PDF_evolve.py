################################
# Cell Aggregate ElectroPoration (CAEP) library
# Author: Poutia Akbari Mistani
# Date: May 4, 2020
# Reference: "On the stochastic electrodynamics of cell aggregate electrodynamics and their anomalous electroporation"
# email: p.a.mistani@gmail.com
###############################
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.special import gamma
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=15, usetex=True)
import scipy.fftpack
import h5py
from scipy import special
#from pyfod.fod import caputo
from mittag_leffler import ml

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class CAEP:
	def __init__(self, epsilon, sigma_e, sigma_c, SL, xmin, xmax, testnum, tf, maxstep =2e-9, eval_points=1000, Nx=2000, phi=0.13):
		self.animate = False
		self.verbose = False
		self.test    = testnum
		self.maxstep = maxstep
		self.eval_point = eval_points
		# physical parameters 
		self.tfinal  = tf   
		self.scaling = 1e-3            
		self.xmin    = xmin                                      # A/mm^2
		self.xmax    = xmax                                      # A/mm^2
		self.x       = np.linspace(self.xmin, self.xmax, Nx)
		self.Delta   = 0.0025 
		self.permittivity_0 = 8.854e-12*self.scaling             #F/mm
		self.E0      = 40000*self.scaling                        # V/mm
		self.sigma_e = sigma_e*self.scaling      #15                       # S/mm
		self.sigma_c = sigma_c*self.scaling      #1                 # S/mm
		self.Cm      = 0.00983*self.scaling**2                      # F/mm^2
		self.SL      = SL*self.scaling**2                       # S/mm^2
		self.h       = 5e-9/self.scaling
		self.sigma_m = self.h*self.SL
		
		
		if self.test != 4:
			self.R       = 7e-6/self.scaling                       # mm
			self.phi     = phi
			self.sigma_t = 2*self.sigma_e + self.sigma_c + self.phi*(self.sigma_e - self.sigma_c)
			self.alpha_b = 3*self.sigma_e*self.sigma_c/(self.Cm*self.R*self.sigma_t)
			self.gamma_b = self.SL/self.Cm + ( self.sigma_e * self.sigma_c * (2 + self.phi))/(self.R*self.Cm*self.sigma_t) 
			self.eta     = 1 + self.SL*self.sigma_t*self.R/((2 + self.phi)*self.sigma_e*self.sigma_c)
			self.tau     = self.Cm*self.sigma_t*self.R/((2 + self.phi)*self.sigma_e*self.sigma_c)
			self.omega   = 2.0*np.pi*1e6
		else:
			self.load_simulation_data()
			self.alpha_b = np.median(self.sim_alphas)
			self.gamma_b = np.median(self.sim_gammas)*1.01
			self.R       = 3*self.sigma_e*self.sigma_c/(self.alpha_b*self.Cm*self.sigma_t) 
			self.eta     = 1 + self.SL*self.sigma_t*self.R/((2 + self.phi)*self.sigma_e*self.sigma_c)
			self.tau     = self.Cm*self.sigma_t*self.R/((2 + self.phi)*self.sigma_e*self.sigma_c)
			self.omega   = 0.0
			
		# experiments
		self.nu_exp    =  7.246803016058461
		self.cc_exp    =  0.8884126275039926
		self.aa_exp    =  0.1635772557705042
		self.lamda_exp = -0.8644983717186614
		self.gamma_p = np.sqrt(self.gamma_b/(self.nu_exp - 0.5))
		self.epsilon = 1.0/np.sqrt(1.0 + self.aa_exp**2/self.lamda_exp**2)
		self.alpha_p = self.alpha_b*self.gamma_p/(self.epsilon*self.gamma_b - self.cc_exp*self.gamma_p**2*np.sqrt(1 - self.epsilon**2))
		self.u_exp   = -self.lamda_exp*self.gamma_p/(self.epsilon*self.alpha_p)
		# if self.test == 4:
		# 	self.plot_simulation_data()


		# initialize the t Student PDF parameters
		self.u0        =  self.sigma_e*self.get_pulse(0)
		self.nu        =  0.5 + self.gamma_b/self.gamma_p**2
		self.lamda     = -self.epsilon*self.alpha_p*self.u0/self.gamma_p
		self.cc        = (self.epsilon*self.alpha_p*self.gamma_b - self.gamma_p*self.alpha_b)/(self.alpha_p*self.gamma_p**2*np.sqrt(1 - self.epsilon**2))
		self.aa        =  self.alpha_p*self.u0*np.sqrt(1 - self.epsilon**2)/self.gamma_p
		self.lam_store = [] #[self.lamda]
		self.aa_store  = [] #[self.aa]
		self.nu_store  = [] #[self.nu]
		self.cc_store  = [] #[self.cc]
		self.u_store   = [] #[self.u0]
		self.W_store   = []
		self.print_params()
		self.W = self.initial_condition()

		# initial conditions
		mu_int           = self.x*self.W
		var_int          = self.x**2*self.W
		self.mu_ini      = integrate.simps(mu_int, self.x) #np.sum(self.x*self.W)*(self.x[1]-self.x[0])       
		self.sigma2_ini  = integrate.simps(var_int, self.x) - self.mu_ini**2 #np.sum(self.x**2*self.W)*(self.x[1]-self.x[0]) - self.mu_ini**2
		#self.mu_ini = self.aa*self.cc/(self.nu - 1) + self.lamda
		#self.sigma2_ini = (self.gamma_p**2*self.mu_ini**2 + 2*self.epsilon*self.gamma_p*self.alpha_p*self.u0*self.mu_ini + self.alpha_p**2*self.u0**2)/(2*(self.gamma_b - self.gamma_p))
		#self.plot()
		#self.Solve()
		
	"""
	initialize by delta distribution around origin
	see Dehghan & Mohammadi 2014
	"""
	def initial_condition(self):
		self.W = (0.5/np.sqrt(np.pi*self.Delta))*np.exp(-self.x**2/(4.0*self.Delta))
		return self.W
		
	def print_params(self, t=0, un=-1):
		print('time= ', t, ' nu=', self.nu, ', c=', self.cc, ', a=', self.aa, ' lambda=', self.lamda, ' u=', un)

	def get_PDF(self):
		self.K = abs(gamma(complex(self.nu, self.cc)))**2/(self.aa*np.sqrt(np.pi)*gamma(self.nu)*gamma(self.nu - 0.5)) 
		self.W = self.K*np.exp(2*self.cc*np.arctan((self.x - self.lamda)/self.aa))/(1 + ((self.x - self.lamda)/self.aa)**2)**self.nu
		return self.W

	def get_pulse(self, t):
		if self.test==0:
			return self.E0
		elif self.test==1: 
			return self.E0*np.sin(self.omega*t)
		elif self.test==2:
			return self.E0*(np.exp(-self.omega*t))
		elif self.test==3:
			if t<1e-6:
				return self.E0
			else:
				return 0.0
		elif self.test==4:
			if t<=1e-6-0.04e-6:
				return self.E0
			else:
				return 0.0
		elif self.test==5:
			return self.E0*np.exp(-6*((t-self.tfinal/3.0)/self.tfinal)**2)
		elif self.test==6:
			if t<1e-6:
				return self.E0*sigmoid(1000*t/self.tfinal)
			else:
				return self.E0*sigmoid(1000*t/self.tfinal)*(np.exp(-40*(t-1e-6)/self.tfinal) )

	def calculate_p_bar(self):
		#p_bar_at_t = np.sum(self.x*self.W)*(self.x[1]-self.x[0])
		integ = self.x*self.W
		p_bar_at_t = integrate.trapz(integ, self.x)
		return p_bar_at_t

	def get_u(self, tnow, pbar):
		# self.p_bar = self.calculate_p_bar()
		E_ext = self.get_pulse(tnow)
		nu = self.sigma_c/self.sigma_e
		
		# case 1. u=sigma_e * E
		#phi = self.phi
		#k = (2 + 3*phi + nu)/(2 + nu + phi*(1-nu)) - 3*phi**2*nu/( (2+ phi)*self.eta*(2 + nu + phi*(1-nu)) )
		#self.u = self.sigma_e*E_ext/k
		# or
		self.u = self.sigma_e*(self.sigma_t*E_ext - nu*self.phi**2*pbar) / ( (2+3*self.phi)*self.sigma_e + self.sigma_c )
		# case 2. u = <J>
		# self.u = self.sigma_e*E_ext*(2+nu+3*phi*nu)/(2+nu+3*phi) + phi*pbar*(1 - (1-nu)*(2+phi - 3*nu*phi**2/(2+3*phi + nu) )/(2+nu+phi*(1-nu)))
		return self.u


	def plot(self):
		plt.figure(figsize=(7,7))
		plt.plot(self.x, self.W, color='k', linewidth=2)
		plt.xlim([self.xmin, self.xmax])
		plt.ylabel(r'$\rm W_s(p_z)$', fontsize=25)
		plt.xlabel(r'$\rm p_z\ [A/mm^2]$', fontsize=25)
		plt.legend(fontsize=20, loc=2, frameon=False)
		plt.tight_layout()
		plt.show()

	def update_params(self, t, mu, s2):
		un = self.get_u(t, mu)
		self.u_store.append(un)
		self.lamda = -self.epsilon*self.alpha_p*un/self.gamma_p
		self.aa    =  self.alpha_p*un*np.sqrt(1 - self.epsilon**2)/self.gamma_p 
		self.nu    = (self.aa**2 + (mu - self.lamda)**2 + 3*s2)/(2*s2)
		self.cc    = (mu - self.lamda)*(self.aa**2 + (mu - self.lamda)**2 + s2)/(2*self.aa*s2)
		self.get_PDF()
		self.print_params(t, un)

	def dmu_sigma2(self, t, y):
		self.update_params(t, y[0], y[1])
		u       =  self.get_u(t, y[0])
		dmu     = -self.gamma_b*y[0] - self.alpha_b*u + self.gamma_p**2*y[0] + self.epsilon*self.alpha_p*self.gamma_p*u
		dsigma2 = -2*(self.gamma_b - self.gamma_p**2)*y[1] + self.gamma_p**2*y[0]**2 + 2*self.epsilon*self.gamma_p*self.alpha_p*u*y[0] + self.alpha_p**2*u**2
		# g3      = -0.1*self.gamma_b
		# dmu    += g3*y[0]**3 + 3*g3*y[0]*y[1]
		# dsigma2+= 6*g3*y[0]**2*y[1]
		return [dmu, dsigma2]

	def jacob(self, t,y):
		return np.array([[-self.gamma_b + 0.5*self.gamma_p**2, 0], [2*self.gamma_p**2*y[0]+2*self.epsilon*self.gamma_p*self.alpha_p*self.get_u(t,y[0]), -4*(self.gamma_b-self.gamma_p**2)*np.sqrt(y[1]) ] ])

	def caputoEuler(self, a, f, y0, tspan):
		"""Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
		D^a y(t) = f(y,t)

		Args:
		a: fractional exponent in the range (0,1)
		f: callable(y,t) returning a numpy array of shape (d,)
			Vector-valued function to define the right hand side of the system
		y0: array of shape (d,) giving the initial state vector y(t==0)
		tspan (array): The sequence of time points for which to solve for y.
			These must be equally spaced, e.g. np.arange(0,10,0.005)
			tspan[0] is the intial time corresponding to the initial state y0.

		Returns:
		y: array, with shape (len(tspan), len(y0))
			With the initial value y0 in the first row

		See also:
		K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
		method
		C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
		Differential Equations
		"""
		d = len(y0)
		N = len(tspan)
		h = (tspan[N-1] - tspan[0])/(N - 1)
		if a<1:
			c = special.rgamma(a) * np.power(h, a) / a
			w = c * np.diff(np.power(np.arange(N), a))
			fhistory = np.zeros((N - 1, d), dtype=type(y0[0]))
			y = np.zeros((N, d), dtype=type(y0[0]))
			y[0] = y0
			for n in range(0, N - 1):
				tn = tspan[n]
				yn = y[n]
				fhistory[n] = f(tn, yn)
				y[n+1] = y0 + np.dot(w[0:n+1], fhistory[n::-1])
			return y
		elif a>1:
			cc = special.gamma(3-a)*np.power(h,a)
			bb = np.diff(np.power(np.arange(N), 2-a))
			fhistory = np.zeros((N - 1, d), dtype=type(y0[0]))
			y = np.zeros((N, d), dtype=type(y0[0]))
			y[0] = y0
			for n in range(0, N - 1):
				tn = tspan[n]
				yn = y[n]
				fhistory[n] = f(tn, yn)
				y[n + 1] = y[n] + cc*fhistory[n] 
				if n>1:
					y[n + 1] += np.dot( bb[:n-1] - bb[1:n], (y[1:n]-y[0:n-1])[::-1] )
			return y 
	


	

	def Solve(self, fraction=1):
		self.t_evaluation = np.linspace(0, self.tfinal, self.eval_point)
		y0 = [self.mu_ini, self.sigma2_ini]
		if fraction!=1:
			self.sol     = self.caputoEuler(fraction, self.dmu_sigma2, y0, self.t_evaluation)
			self.times   = self.t_evaluation
			self.mus     = self.sol[:,0]
			self.sigma2s = self.sol[:,1]
		else:
			self.sol     = solve_ivp(self.dmu_sigma2, [0, self.tfinal], y0, method='LSODA', rtol=1e-15, jac=self.jacob, max_step=self.maxstep, t_eval=self.t_evaluation) 
			self.times   = self.sol.t
			self.mus     = self.sol.y[0]
			self.sigma2s = self.sol.y[1]


		self.Es = np.array([self.get_pulse(t) for t in self.times])
		

		if self.test==4:
			a = 1.01
			fractional_pz1  = 1 - ml(-(self.gamma_b-self.gamma_p**2)*(self.times)**a, a)
			fractional_pz1 += -( 1 - ml(-(self.gamma_b-self.gamma_p**2)*(abs(self.times - 1e-6+0.04e-6))**a, a) ) * np.heaviside(self.times - 1e-6+0.04e-6, 0)
			fractional_pz1 *= self.get_u(0,0)*(self.epsilon*self.gamma_p*self.alpha_p - self.alpha_b)/(self.gamma_b - self.gamma_p**2)

			# a = 0.99
			# fractional_pz2  = 1 - ml(-(self.gamma_b-self.gamma_p**2)*(self.times)**a, a)
			# fractional_pz2 += -( 1 - ml(-(self.gamma_b-self.gamma_p**2)*(abs(self.times - 1e-6))**a, a) ) * np.heaviside(self.times - 1e-6, 0)
			# fractional_pz2 *= self.get_u(0,0)*(self.epsilon*self.gamma_p*self.alpha_p - self.alpha_b)/(self.gamma_b - self.gamma_p**2)

			fig, ax = plt.subplots(1, 2, figsize=(15,7))
			# ax[0].plot(1e6*self.times, abs(fractional_pz2), color='b', linewidth=1 , linestyle='-.', label=r'$\rm model,\ \alpha=0.99$'))
			ax[0].plot(1e6*self.times, -self.mus, 'r', linestyle='-', linewidth=1, label=r'$\rm model,\ \alpha=1$')
			ax[0].plot(1e6*self.times, abs(fractional_pz1), color='g', linewidth=2, linestyle='--', label=r'$\rm model,\ \alpha=1.01$')
			
			ax[0].plot(self.sim_time[:49], np.mean(self.sim_px[:49,self.permitted], axis=1), color='purple', linestyle=':', linewidth=2, label=r'$\rm p_x\ simulation$')
			ax[0].plot(self.sim_time[:49], np.mean(self.sim_py[:49,self.permitted], axis=1), color='m', linestyle='-.', linewidth=2, label=r'$\rm p_y\ simulation$')
			ax[0].scatter(self.sim_time[:49], np.mean(self.sim_pz[:49,self.permitted], axis=1), color='k', marker='p', s=20, label=r'$\rm p_z\ simulation$')
			
			
			ax[1].plot(1e6*self.times, self.sigma2s, 'r', linestyle='-', linewidth=1, label=r'$\rm model,\ \alpha=1$')
			ax[1].plot(self.sim_time[:49], np.var(self.sim_px[:49,self.permitted], axis=1), color='purple', linestyle=':', linewidth=2, label=r'$\rm p_x\ simulation$')
			ax[1].plot(self.sim_time[:49], np.var(self.sim_py[:49,self.permitted], axis=1), color='m', linestyle='-.', linewidth=2, label=r'$\rm p_y\ simulation$')
			ax[1].scatter(self.sim_time[:49], self.m2z[:49], color='k', marker='p', s=20, label=r'$\rm p_z\ simulation$')
			ax[0].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			ax[1].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			ax[0].set_ylabel(r'$\rm \vert \mu\vert \ [A/mm^2]$', fontsize=25)
			ax[1].set_ylabel(r'$\rm \sigma^2\ [A^2/mm^4]$', fontsize=25)
			ax[0].set_ylim([0, 1.0])
			ax[1].yaxis.set_ticks_position("right")
			ax[1].yaxis.set_label_position("right") 
			ax[0].legend(fontsize=15, frameon=True)
			ax[1].legend(fontsize=15, frameon=True)
			plt.tight_layout()
			plt.show()
			# data_sim = np.loadtxt("data_time_mean_variance_ps.dat")
			# fig, ax = plt.subplots(1, 2, figsize=(15,7))
			# ax[0].plot(1e6*self.times, -self.mus, 'r', linestyle='-', linewidth=2, label=r'$\rm p_z\ model$')
			# ax[0].plot(data_sim[:,0], data_sim[:,1], color='g', linestyle=':', linewidth=1, label=r'$\rm p_x\ simulation$')
			# ax[0].plot(data_sim[:,0], data_sim[:,2], color='b', linestyle='-.', linewidth=1, label=r'$\rm p_y\ simulation$')
			# ax[0].scatter(data_sim[:,0], data_sim[:,3], color='k', marker='o', s=10, label=r'$\rm p_z\ simulation$')
			# ax[1].plot(1e6*self.times, self.sigma2s, 'r', linestyle='-', linewidth=2, label=r'$\rm p_z\ model$')
			# ax[1].plot(data_sim[:,0], data_sim[:,4], color='g', linestyle=':', linewidth=1, label=r'$\rm p_x\ simulation$')
			# ax[1].plot(data_sim[:,0], data_sim[:,5], color='b', linestyle='-.', linewidth=1, label=r'$\rm p_y\ simulation$')
			# ax[1].scatter(data_sim[:,0], data_sim[:,6], color='k', marker='o', s=10, label=r'$\rm p_z\ simulation$')
			# ax[0].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			# ax[1].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			# ax[0].set_ylabel(r'$\rm \vert \mu\vert \ [A/mm^2]$', fontsize=25)
			# ax[1].set_ylabel(r'$\rm \sigma^2\ [A^2/mm^4]$', fontsize=25)
			# ax[0].set_ylim([0, 1.0])
			# ax[1].yaxis.set_ticks_position("right")
			# ax[1].yaxis.set_label_position("right") 
			# ax[0].legend(fontsize=15, frameon=True)
			# ax[1].legend(fontsize=15, frameon=True)
			# plt.tight_layout()
			# plt.show()
		else:
			fig, ax = plt.subplots(1, 2, figsize=(15,7))
			ax[0].plot(1e6*self.times, -self.mus, 'r', linestyle='-', linewidth=2)
			ax[1].plot(1e6*self.times, self.sigma2s, 'r', linestyle='-', linewidth=2)
			ax[0].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			ax[1].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			ax[0].set_ylabel(r'$\rm \vert \mu\vert \ [A/mm^2]$', fontsize=25)
			ax[1].set_ylabel(r'$\rm \sigma^2\ [A^2/mm^4]$', fontsize=25)
			ax[0].set_ylim([0, 1.0])
			ax[1].yaxis.set_ticks_position("right")
			ax[1].yaxis.set_label_position("right") 
			plt.tight_layout()
			plt.show()

		if self.verbose:
			if self.animate:
				plt.ion()
				plt.figure(figsize=(7,7))
			counter = 0
			for tt, mu, s2 in zip(self.times, self.mus, self.sigma2s):
				self.update_params(tt, mu, s2)
				self.lam_store.append(self.lamda)
				self.aa_store.append(self.aa)
				self.nu_store.append(self.nu)
				self.cc_store.append(self.cc)
				self.W_store.append(self.W)
				if self.animate:
					plt.plot(self.x, self.W, color='k', linewidth=2)
					plt.xlim([self.xmin, self.xmax])
					plt.ylim([0, 5])
					plt.ylabel(r'$\rm W_s(p_z)$', fontsize=25)
					plt.xlabel(r'$\rm p_z\ [A/mm^2]$', fontsize=25)
					plt.legend(fontsize=20, loc=2, frameon=False)
					plt.tight_layout()
					plt.draw()
					#plt.savefig("./movie/snap_"+str(counter).zfill(4)+".png")
					plt.pause(.001)
					plt.clf()
				counter += 1
			if self.animate:
				plt.close()
			plt.figure(figsize=(7,7))
			plt.plot(1e6*self.times, self.aa_store, linestyle='-', label=r'$\rm a$')
			plt.plot(1e6*self.times, self.nu_store, linestyle='-.', label=r'$\rm \nu$')
			plt.plot(1e6*self.times, self.cc_store, linestyle=':', label=r'$\rm c$')
			plt.plot(1e6*self.times, self.lam_store, linestyle='--', label=r'$\rm \lambda$')
			plt.ylim([-1,10])
			plt.xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			plt.legend(frameon=False, fontsize=20)
			plt.tight_layout()
			plt.show()

			self.W_store = np.array(self.W_store)
			self.aa_store = np.array(self.aa_store)
			self.nu_store = np.array(self.nu_store)
			self.cc_store = np.array(self.cc_store)
			# moms = []
			# moms.append( np.array(len(self.mus)*[1]) )
			# moms.append( np.array(len(self.mus)*[0]) )
			# for n in range(2,11):
			# 	tmp = ( (n-1)*self.aa_store/( (self.nu_store-1)**2 * (2*(self.nu_store-1) - (n-1)) ) ) * ( (self.nu_store-1)*self.cc_store*moms[n-1] + self.aa_store*moms[n-2]*((self.nu_store-1)**2 + self.cc_store**2) ) 
			# 	moms.append(np.array(tmp))
			# pdb.set_trace()






	def evolution_pdf(self):
		X, T = np.meshgrid(-self.x, 1e6*self.times)
		# self.W_store = np.nan_to_num(sep.W_store)
		# self.W_store[self.W_store>10]=10
		cmap = plt.cm.plasma
		cmap.set_under(color='w')
		plt.figure(figsize=(7,7))
		plt.pcolor(T, X, self.W_store, cmap=cmap) #, vmin=0.2)
		plt.xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		plt.ylabel(r'$\rm \vert p_z\vert \ [A/mm^2]$', fontsize=25)
		plt.ylim([0, 1.3])
		plt.tight_layout()
		plt.show()



	def time_domain_analysis(self):
		mus  = self.mus 
		var_p= self.sigma2s
		Es   = self.Es 
		times= self.times
		nu   = self.sigma_c/self.sigma_e
		area = 1 # mm^2
		H    = 1 #mm

		E_t = (self.sigma_t*Es - nu*self.phi**2*mus)/((2+3*self.phi)*self.sigma_e + self.sigma_c)
		s_t = -3*((self.sigma_e - self.sigma_c)/self.sigma_t) * (self.sigma_e*E_t + (2+ self.phi)*mus/3)
		J_t = self.sigma_e*Es + self.phi*s_t + self.phi*mus*(1-self.sigma_m/self.sigma_e) 
		sigma_bar_per_sigma_e_t = J_t/(self.sigma_e*Es)
		Resistance = H*Es/(area*J_t)

		fig, ax = plt.subplots(2, figsize=(7,7))
		ax[0].plot(1e6*times, Es/40.0, linewidth=1, color='b', linestyle='-.', label=r'$\rm E_{ext}/40\ [kV/m]$')
		ax[0].plot(1e6*times, sigma_bar_per_sigma_e_t, linewidth=1, color='r', linestyle=':', label=r'$\rm  \bar{\sigma}(t)/\sigma_e$')
		ax[0].plot(1e6*times, Resistance/1000.0, linewidth=1, color='k', linestyle='-', label=r'$\rm R(t)\ [k\Omega]$')
		ymin0 = np.min([np.min(Es/40.0), np.min(sigma_bar_per_sigma_e_t), np.min(Resistance/1000)])
		ymax0 = np.max([np.max(Es/40.0), np.max(sigma_bar_per_sigma_e_t), np.max(Resistance/1000)])
		ax[0].set_ylim([1.1*ymin0, 1.1*ymax0])
		# ax[0].set_ylim([0,1.1])
		ax[1].plot(1e6*times, s_t, linewidth=1, color='b', linestyle='-.', label=r'$\rm s(t)\ [A/mm^2]$')
		ax[1].plot(1e6*times, mus, linewidth=1, color='g', linestyle='--', label=r'$\rm p(t)\ [A/mm^2]$')
		ax[1].plot(1e6*times, var_p*10, linewidth=1, color='r', linestyle=':', label=r'$\rm var(p)(t)\times 10\ [A^2/mm^4]$')
		ax[1].plot(1e6*times, J_t*area, linewidth=1, color='k', linestyle='-', label=r'$\rm I(t)\ [A]$')
		ymin1 = np.min([np.min(s_t), np.min(mus), np.min(var_p*10), np.min(J_t*area)])
		ymax1 = np.max([np.max(s_t), np.max(mus), np.max(var_p*10), np.max(J_t*area)])
		ax[1].set_ylim([1.1*ymin1, 1.1*ymax1])
		ax[0].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		ax[1].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		ax[0].set_ylabel(r'$\rm value$', fontsize=25)
		ax[1].set_ylabel(r'$\rm value$', fontsize=25)
		ax[0].legend(fontsize=12, frameon=False)
		ax[1].legend(fontsize=12, frameon=False)
		plt.tight_layout()
		plt.show()


	def fourier_analysis(self):
		indices = self.times > 0
		mus = self.mus[indices]
		Es  = self.Es[indices]
		times = self.times[indices]

		N = len(times)
		T = times[1] - times[0]

		yf = scipy.fftpack.fft(mus)
		self.Pf = (2.0/N) * yf[:N//2]
		self.yf = abs(self.Pf) 
		self.xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
		
		Ef = scipy.fftpack.fft(Es)
		self.Eextf = (2.0/N) * Ef[:N//2]
		self.Ef = (2.0/N) * np.abs(Ef[:N//2])

		if self.test==1:
			self.alpha_p = -3/( (2+self.phi)*(self.eta + complex(0,1)*self.omega*self.tau) )
			self.alpha_s = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*self.omega*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*self.omega*self.tau))
		else:
			self.alpha_p = -3/(2+self.phi)/(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau )
			self.alpha_s = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*2*np.pi*self.xf*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau))
		

		self.alpha_e = (2*self.sigma_e + (1 + self.phi*self.alpha_p)*self.sigma_c)/self.sigma_t
		self.k = self.alpha_e - self.phi*self.alpha_p - self.phi*self.sigma_e*self.alpha_s/(self.sigma_e - self.sigma_c)


		self.chi_s = self.alpha_s*self.phi / self.k
		self.chi_p = self.alpha_p*self.phi / self.k
		Eavg = self.Eextf
		area = 1.0 # mm^2
		H    = 1.0    # mm
		Ze   = self.sigma_e*area/H
		self.Znormal = (self.sigma_e*self.Eextf)/(self.sigma_e*self.Eextf + self.phi*self.Pf*(1-self.sigma_m/self.sigma_e) + self.phi*self.Pf*self.chi_s/self.chi_p ) 
		
		self.permittivity = self.sigma_e*(self.chi_p*(1-self.sigma_m/self.sigma_e) + self.chi_s)
		self.permittivity /= complex(0,1)*2*np.pi*self.xf*self.permittivity_0



		plt.figure(figsize=(7,7))
		plt.plot(self.mus, self.Es, color='k', linewidth=1)
		plt.xlabel(r"$\rm \mu \ [A/mm^2]$", fontsize=25)
		plt.ylabel(r"$\rm E_{ext}\ [kV/m]$", fontsize=25)
		plt.ylim([1.1*np.min(self.Es), 1.1*np.max(self.Es)])
		plt.xlim([1.1*np.min(self.mus), 1.1*np.max(self.mus)])
		plt.tight_layout()
		plt.show()

		fig, axes = plt.subplots(2, figsize=(7,7))
		axes[0].plot(self.xf, np.real(self.Pf), linewidth=1, color='k')
		axes[1].plot(self.xf, np.imag(self.Pf), linewidth=1, color='k')
		axes[0].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[1].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[0].set_ylabel(r'$\rm Re(\tilde{p})\ [A/mm^2]$', fontsize=25)
		axes[1].set_ylabel(r'$\rm Im(\tilde{p})\ [A/mm^2]$', fontsize=25)
		axes[0].set_xscale('log')
		axes[1].set_xscale('log')
		plt.tight_layout()
		plt.show()

		plt.figure(figsize=(7,7))
		plt.plot(self.xf, self.yf, linewidth=2, color='k')
		plt.scatter(self.xf, self.yf, s=5, color='r')
		plt.xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		plt.ylabel(r'$\rm \vert \tilde{p} \vert \ [A/mm^2]$', fontsize=25)
		plt.yscale('log')
		plt.xscale('log')
		plt.tight_layout()
		plt.show()

		fig, axes = plt.subplots(2, figsize=(7,7))
		axes[0].plot(self.xf, np.real(self.permittivity), linewidth=2, color='k')
		axes[1].plot(self.xf, -np.imag(self.permittivity), linewidth=2, color='k')
		axes[0].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[1].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[0].set_ylabel(r"$\rm \epsilon'$", fontsize=25)
		axes[1].set_ylabel(r"$\rm \epsilon''$", fontsize=25)
		axes[0].set_xscale('log')
		axes[1].set_xscale('log')
		axes[0].set_yscale('log')
		axes[1].set_yscale('log')
		plt.tight_layout()
		plt.show()
		plt.figure(figsize=(7,7))
		plt.plot(np.real(self.permittivity), -np.imag(self.permittivity), linewidth=2, color='k')
		plt.xlabel(r"$\rm \epsilon'$", fontsize=25)
		plt.ylabel(r"$\rm \epsilon''$", fontsize=25)
		plt.tight_layout()
		plt.show()

		fig, axes = plt.subplots(2, figsize=(7,7))
		axes[0].plot(self.xf, np.real(self.Znormal), linewidth=1, color='k')
		axes[1].plot(self.xf, np.imag(self.Znormal), linewidth=1, color='k')
		axes[0].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[1].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[0].set_ylabel(r'$\rm Re(Z)/Z_e$', fontsize=25)
		axes[1].set_ylabel(r'$\rm Im(Z)/Z_e$', fontsize=25)
		axes[0].set_xscale('log')
		axes[1].set_xscale('log')
		plt.tight_layout()
		plt.show()

		fig, axes = plt.subplots(2, figsize=(7,7))
		axes[0].plot(self.xf, np.abs(self.Znormal), linewidth=1, color='k')
		axes[1].plot(self.xf, np.angle(self.Znormal), linewidth=1, color='k')
		axes[0].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[1].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[0].set_ylabel(r'$\rm \vert Z\vert/Z_e$', fontsize=25)
		axes[1].set_ylabel(r'$\rm \angle Z\ [rad]$', fontsize=25)
		axes[0].set_xscale('log')
		axes[1].set_xscale('log')
		plt.tight_layout()
		plt.show()

		# impedance locus: -Im(Z) vs. Re(Z) => if semicircle arc below x-axis it is anomalous 
		Zmax = np.max(np.real(self.Znormal)) + 0.1
		Zmin = np.min(np.real(self.Znormal)) - 0.1
		plt.figure(figsize=(7,7))
		plt.plot(np.real(self.Znormal), -np.imag(self.Znormal), linewidth=1, color='k')
		plt.xlabel(r'$\rm Re(Z) /Z_e$', fontsize=25)
		plt.ylabel(r'$\rm -Im(Z)/Z_e$', fontsize=25)
		plt.xlim([Zmin, Zmax])
		plt.ylim([0, Zmax-Zmin])
		plt.tight_layout()
		plt.show()

	def relaxation(self):
		if self.test==3:
			mus = abs(self.mus[self.times>1e-6])
			times = self.times[self.times>1e-6]
			plt.figure(figsize=(7,7))
			plt.plot(1e6*times, mus, linewidth=1, color='k')
			plt.xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
			plt.ylabel(r'$\rm \vert \mu\vert \ [A/mm^2]$', fontsize=25)
			plt.yscale('log')
			plt.tight_layout()
			plt.show()
		else:
			pass

	def statistics(self):
		phis = np.array([0.1, 0.15, 0.25, 0.35, 0.5, 0.65])
		cols = ['darkblue', 'b', 'c', 'g', 'm', 'red']
		lines = ['--', ':', '-', '-.', '--', '-'] 
		R = 7e-6
		Cm = 0.01
		SL = 1.9
		sigma_e = 1.3
		sigma_c = 0.6
		h = 5e-9
		sigma_m = SL*h
		
		chi_p_store = []
		chi_s_store = []
		E_e_Eext_store = []
		E_Eext_store = []
		for phi in phis:
			sigma_t = 2*sigma_e + sigma_c + phi*(sigma_e - sigma_c)
			tau = Cm*sigma_t*R/((2 + phi)*sigma_e*sigma_c)
			eta = 1 + SL*sigma_t*R/((2 + phi)*sigma_e*sigma_c)
			freqs = np.logspace(4, 8, 200)
			alpha_p = -3/(2+phi)/(eta + complex(0,1)*2*np.pi*freqs*tau )
			alpha_s = -3*(sigma_e - sigma_c)*(eta - 1 + complex(0,1)*2*np.pi*freqs*tau )/(sigma_t*(eta + complex(0,1)*2*np.pi*freqs*tau))
			alpha_e = (2*sigma_e + (1 + phi*alpha_p)*sigma_c)/sigma_t
			k = alpha_e - phi*alpha_p - phi*sigma_e*alpha_s/(sigma_e - sigma_c)
			chi_p = alpha_p*phi/k
			chi_s = alpha_s*phi/k
			E_e_Eext = alpha_e/(k*(1-phi))
			E_Eext = 1.0/k
			chi_p_store.append(chi_p)
			chi_s_store.append(chi_s)
			E_e_Eext_store.append(E_e_Eext)
			E_Eext_store.append(E_Eext)

		chi_p_store = np.array(chi_p_store)
		chi_s_store = np.array(chi_s_store)
		E_e_Eext_store = np.array(E_e_Eext_store)
		E_Eext_store = np.array(E_Eext_store)

		phi_theta_1 = (chi_p_store*(1 - sigma_m/sigma_e) + chi_s_store)/3.0 
		sigma_eff_per_sigma_e = 1 + 3*phi_theta_1 + 3*phi_theta_1**2/(1 - phi_theta_1)


		plt.figure(figsize=(7,7))
		for chip, col, ln, phi in zip(chi_p_store, cols, lines, phis): plt.plot(freqs, abs(chip), linewidth=1, c=col, linestyle=ln, label=r'$\rm \phi=$'+str(phi))
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		plt.ylabel(r'$\rm \vert \chi_p\vert$', fontsize=25)
		plt.ylim([0.01,1])
		plt.tight_layout()
		plt.legend(fontsize=20, frameon=False)
		plt.show()
		
		plt.figure(figsize=(7,7))
		for Evg, col, ln, phi in zip(E_Eext_store, cols, lines, phis): plt.plot(freqs, abs(Evg), linewidth=1, c=col, linestyle=ln, label=r'$\rm \phi=$'+str(phi))
		plt.xscale('log')
		plt.xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		plt.ylabel(r'$\rm \vert E\vert/\vert E_{ext}\vert$', fontsize=25)
		plt.ylim([-0.05,1.1])
		plt.legend(fontsize=20, frameon=False, loc=4)
		plt.tight_layout()
		plt.show()

		plt.figure(figsize=(7,7))
		for Ee, col, ln, phi in zip(E_e_Eext_store, cols, lines, phis): plt.plot(freqs, abs(Ee), linewidth=1, c=col, linestyle=ln, label=r'$\rm \phi=$'+str(phi))
		plt.xscale('log')
		plt.xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		plt.ylabel(r'$\rm \vert E_{e}\vert/\vert E_{ext}\vert$', fontsize=25)
		plt.ylim([0.9,2])
		plt.legend(fontsize=20, frameon=False)#, loc=4)
		plt.tight_layout()
		plt.show()

		plt.figure(figsize=(7,7))
		for siff, col, ln, phi in zip(sigma_eff_per_sigma_e, cols, lines, phis): plt.plot(freqs, abs(siff), linewidth=1, c=col, linestyle=ln, label=r'$\rm \phi=$'+str(phi))
		plt.xscale('log')
		plt.xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		plt.ylabel(r'$\rm \vert \bar{\sigma}/ \sigma_e \vert$', fontsize=25)
		plt.ylim([-0.05,1.1])
		plt.legend(fontsize=20, frameon=False, loc=3)
		plt.tight_layout()
		plt.show()
		

		## Now choose 6 frequencies and vary phi continuously
		chi_p_store_2 = []
		chi_s_store_2 = []
		freqs_2 = [1e4, 1e5, 1e6, 2e6, 1e7, 1e8]
		labels = [r'$10^4\ [Hz]$', r'$10^5\ [Hz]$', r'$10^6\ [Hz]$', r'$2\times 10^6\ [Hz]$', r'$10^7\ [Hz]$', r'$10^8\ [Hz]$']
		phis = np.linspace(0, 1, 100)
		for freq in freqs_2:
			sigma_t = 2*sigma_e + sigma_c + phis*(sigma_e - sigma_c)
			tau = Cm*sigma_t*R/((2 + phis)*sigma_e*sigma_c)
			eta = 1 + SL*sigma_t*R/((2 + phis)*sigma_e*sigma_c)
			alpha_p = -3/(2+phis)/(eta + complex(0,1)*2*np.pi*freq*tau )
			alpha_s = -3*(sigma_e - sigma_c)*(eta - 1 + complex(0,1)*2*np.pi*freq*tau )/(sigma_t*(eta + complex(0,1)*2*np.pi*freq*tau))
			alpha_e = (2*sigma_e + (1 + phis*alpha_p)*sigma_c)/sigma_t
			k = alpha_e - phis*alpha_p - phis*sigma_e*alpha_s/(sigma_e - sigma_c)
			chi_p = alpha_p*phis/k
			chi_s = alpha_s*phis/k
			chi_p_store_2.append(chi_p)
			chi_s_store_2.append(chi_s)
		chi_p_store_2 = np.array(chi_p_store_2)
		chi_s_store_2 = np.array(chi_s_store_2)

		phi_theta_1 = (chi_p_store_2*(1 - sigma_m/sigma_e) + chi_s_store_2)/3.0 
		sigma_eff_per_sigma_e_2 = 1 + 3*phi_theta_1 + 3*phi_theta_1**2/(1-phi_theta_1)

		plt.figure(figsize=(7,7))
		for siff, col, ln, lab in zip(sigma_eff_per_sigma_e_2, cols, lines, labels): plt.plot(phis, abs(siff), linewidth=1, c=col, linestyle=ln, label=r'$\rm f=$'+lab)
		plt.xlabel(r'$\rm \phi$', fontsize=25)
		plt.ylabel(r'$\rm \vert \bar{\sigma}/ \sigma_e \vert$', fontsize=25)
		plt.ylim([-0.05,1.1])
		plt.legend(fontsize=20, frameon=False, loc=3)
		plt.tight_layout()
		plt.show()


	def load_simulation_data(self, plot=False):
		data = np.loadtxt("algam.dat")
		# self.sim_alphas  = data[:,0]
		# self.sim_gammas  = data[:,1]
		self.sim_radii   = data[:,2]/self.scaling
		self.sim_volumes = data[:,3]/self.scaling**3
		self.phi = self.sim_volumes.sum()/4.0**3       # box=2*2*2 mm^3
		self.sigma_t = 2*self.sigma_e + self.sigma_c + self.phi*(self.sigma_e - self.sigma_c)

		self.sim_alphas = 3*self.sigma_e*self.sigma_c/(self.Cm*self.sim_radii*self.sigma_t)
		self.sim_gammas = self.SL/self.Cm + self.sigma_c*self.sigma_e*(2+self.phi)/(self.Cm*self.sim_radii*self.sigma_t)

		hf = h5py.File('data_pxyz.h5', 'r')
		self.permitted = np.array(hf.get('permitted'))
		self.sim_px = np.array(hf.get('px'))
		self.sim_py = np.array(hf.get('py'))
		self.sim_pz = np.array(hf.get('pz'))
		self.sim_time = np.array(hf.get('t')) + 0.04
		hf.close()
		# statistical moments about average
		m1z = []
		m2z = []
		m3z = []
		m4z = []
		m5z = []
		m6z = []
		m7z = []
		m8z = []
		m9z = []
		m10z = []
		for snap in range(len(self.sim_pz)):
			dz = self.sim_pz[snap] - np.mean(self.sim_pz[snap])
			m1z.append(np.mean(self.sim_pz[snap]))
			m2z.append(np.mean(dz**2))
			m3z.append(np.mean(dz**3))
			m4z.append(np.mean(dz**4))
			m5z.append(np.mean(dz**5))
			m6z.append(np.mean(dz**6))
			m7z.append(np.mean(dz**7))
			m8z.append(np.mean(dz**8))
			m9z.append(np.mean(dz**9))
			m10z.append(np.mean(dz**10))
		self.m1z = np.array(m1z)
		self.m2z = np.array(m2z)
		self.m3z = np.array(m3z)
		self.m4z = np.array(m4z)
		self.m5z = np.array(m5z)
		self.m6z = np.array(m6z)
		self.m7z = np.array(m7z)
		self.m8z = np.array(m8z)
		self.m9z = np.array(m9z)
		self.m10z = np.array(m10z)


	def plot_simulation_data(self):
		a = 1.008
		fractional_pz = self.get_u(0,0)*(1-ml(-(self.gamma_b-self.gamma_p**2)*(1e-6*self.sim_time[:23])**a, a))*(self.epsilon*self.gamma_p*self.alpha_p - self.alpha_b)/(self.gamma_b - self.gamma_p**2)
		fig, ax = plt.subplots(1, 2, figsize=(15,7))
		ax[0].plot(self.sim_time[:49], np.mean(self.sim_px[:49,self.permitted], axis=1), 'r', linestyle='-.', label='data, x')
		ax[0].plot(self.sim_time[:49], np.mean(self.sim_py[:49,self.permitted], axis=1), 'g', linestyle='--', label='data, y')
		ax[0].plot(self.sim_time[:49], np.mean(self.sim_pz[:49,self.permitted], axis=1), 'b', linestyle='-', label='data, z')
		ax[0].plot(self.sim_time[:23], abs(fractional_pz), color='r', linestyle='--' )
		ax[1].plot(self.sim_time[:49], np.var(self.sim_px[:49,self.permitted], axis=1), 'r', linestyle='-.', label='data, x')
		ax[1].plot(self.sim_time[:49], np.var(self.sim_py[:49,self.permitted], axis=1), 'g', linestyle='--', label='data, y')
		ax[1].plot(self.sim_time[:49], self.m2z[:49], 'b', linestyle='-', label='data, z')
		ax[0].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		ax[1].set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		ax[0].set_ylabel(r'$\rm \vert \mu\vert \ [A/mm^2]$', fontsize=25)
		ax[1].set_ylabel(r'$\rm \sigma^2 \ [A^2/mm^4]$', fontsize=25)
		ax[1].yaxis.set_ticks_position("right")
		ax[1].yaxis.set_label_position("right")
		ax[0].set_ylim([0,0.9])
		ax[1].set_ylim([0, 0.006])
		ax[0].legend(fontsize=20, frameon=False)
		ax[1].legend(fontsize=20, frameon=False)
		plt.tight_layout()
		plt.show()		

		fig, ax = plt.subplots(1,1,figsize=(7,7))
		ax.hist(self.sim_alphas/np.mean(self.sim_alphas), 100, histtype='step', color='r', hatch='/', label=r'$\rm \alpha/\bar{\alpha}$', density=True)
		ax.hist(self.sim_gammas/np.mean(self.sim_gammas), 100, histtype='step', color='b', hatch='-', label=r'$\rm \gamma/\bar{\gamma}$', density=True)
		ax.set_ylabel(r"$\rm PDF$", fontsize=25)
		ax.set_xlabel(r"$\rm value$", fontsize=25)
		# ax.set_xscale('log')
		ax.set_yscale('log')
		ax.legend(fontsize=20, frameon=False)
		plt.tight_layout()
		plt.show()

		fig, ax = plt.subplots(1,1,figsize=(7,7))
		ax.plot(self.sim_time, self.m2z, label=r'$\rm m_2$', color='b', linewidth=1)
		ax.plot(self.sim_time, self.m3z, label=r'$\rm m_3$', color='r', linewidth=1)
		ax.plot(self.sim_time, self.m4z, label=r'$\rm m_4$', color='c', linewidth=1)
		ax.plot(self.sim_time, self.m5z, label=r'$\rm m_5$', color='m', linewidth=1)
		ax.plot(self.sim_time, self.m6z, label=r'$\rm m_6$', color='k', linewidth=1)
		ax.set_ylim([-0.01,0.025])
		ax.set_xlim([0, 4.1])
		ax.set_xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		ax.legend(fontsize=20, frameon=False)
		plt.tight_layout()
		plt.show()

		m1max = np.max(abs(self.m1z))
		m2max = np.max(abs(self.m2z))
		m3max = np.max(abs(self.m3z))
		m4max = np.max(abs(self.m4z))
		m5max = np.max(abs(self.m5z))
		m6max = np.max(abs(self.m6z))
		m7max = np.max(abs(self.m7z))
		m8max = np.max(abs(self.m8z))
		m9max = np.max(abs(self.m9z))
		m10max = np.max(abs(self.m10z))
		mmaxs = [m1max, m2max, m3max, m4max, m5max, m6max, m7max, m8max, m9max, m10max]
		ks = range(1,11)
		fig, ax = plt.subplots(1,1,figsize=(7,7))
		ax.plot(ks, mmaxs, color='r', linewidth=1, linestyle='--')
		ax.scatter(ks, mmaxs, color='k', marker='x', s=50)
		ax.set_xlabel(r'$\rm k$', fontsize=25)
		ax.set_ylabel(r'$\rm max(\vert m_k\vert)$', fontsize=25)
		plt.tight_layout()
		plt.show()



testnum    =  4           # 6: smoothed step function, 5: Gaussian, 4: sharp step pulse, 3: relaxation test
tf         =  2.0e-6	      # [s]
eps        =  0.982       
xmin       = -5.0
xmax       =  5.0
nx         =  2000
t_samples  =  1e3 #1e5
max_t_step =  1e-9
sigma_e    =  15
sigma_c    =  1
SL         =  1.9
phi        =  0.0025 #0.13 #*(4*np.pi/3.0)*(5e-4)**3/(1e-9)
sep        =  CAEP(eps, sigma_e, sigma_c, SL, xmin, xmax, testnum, tf, max_t_step, t_samples, nx, phi)

sep.Solve(1.1)

# sep.relaxation()
# sep.evolution_pdf()    # must set verbose=True in Solve.
# sep.statistics()
# sep.time_domain_analysis()
# sep.fourier_analysis()

# pdb.set_trace()

