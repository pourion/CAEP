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

class CAEP:
	def __init__(self, epsilon, xmin, xmax):
		self.sinusoid = False
		# physical parameters 
		self.tfinal  = 1e-6   
		self.scaling = 1e-3            
		self.xmin    = xmin                                      # A/mm^2
		self.xmax    = xmax                                      # A/mm^2
		self.x       = np.linspace(self.xmin, self.xmax, 2000)
		self.E0      = 40000*self.scaling                        # V/mm
		self.sigma_e = 15*self.scaling                           # S/mm
		self.sigma_c = 1*self.scaling                            # S/mm
		self.Cm      = 0.01*self.scaling**2                      # F/mm^2
		self.SL      = 1.9*self.scaling**2                       # S/mm^2
		self.R       = 7.0e-6/self.scaling                       # mm
		self.phi     = 0.13

		self.sigma_t = 2*self.sigma_e + self.sigma_c + self.phi*(self.sigma_e - self.sigma_c)
		self.alpha_b = 3*self.sigma_e*self.sigma_c/(self.Cm*self.R*self.sigma_t)
		self.gamma_b = self.SL/self.Cm + (self.sigma_e * self.sigma_c * (2 + self.phi))/(self.R*self.Cm*self.sigma_t)
		# stochastic parameters
		self.gamma_p = np.sqrt(self.gamma_b/7.0)        # noise in gamma
		self.alpha_p = np.sqrt(self.alpha_b/3.0)        # noise in alpha
		self.epsilon = epsilon                          # correlation between alpha and gamma \in [-1, 1]

		# initialize the t Student PDF parameters
		self.u0        = self.sigma_e*self.get_pulse(0)
		self.nu        =  0.5 + self.gamma_b/self.gamma_p**2
		self.lamda     = -self.epsilon*self.alpha_p*self.u0/self.gamma_p
		self.cc        = (self.epsilon*self.alpha_p*self.gamma_b - self.gamma_p*self.alpha_b)/(self.alpha_p*self.gamma_p**2*np.sqrt(1 - self.epsilon**2))
		self.aa        =  self.alpha_p*self.u0*np.sqrt(1 - self.epsilon**2)/self.gamma_p
		self.lam_store = [] #[self.lamda]
		self.aa_store  = [] #[self.aa]
		self.nu_store  = [] #[self.nu]
		self.cc_store  = [] #[self.cc]
		self.u_store   = [] #[self.u0]
		self.print_params()
		# initialize W
		self.W = self.get_PDF()
		# initial conditions
		self.mu_ini      = np.sum(self.x*self.W)*(self.x[1]-self.x[0])       
		self.sigma2_ini  = np.sum(self.x**2*self.W)*(self.x[1]-self.x[0]) - self.mu_ini**2
		#self.mu_ini = self.aa*self.cc/(self.nu - 1) + self.lamda
		#self.sigma2_ini = (self.gamma_p**2*self.mu_ini**2 + 2*self.epsilon*self.gamma_p*self.alpha_p*self.u0*self.mu_ini + self.alpha_p**2*self.u0**2)/(2*(self.gamma_b - self.gamma_p))
		#self.plot()
		self.Solve()
		
		
	def print_params(self, t=0):
		print('time= ', t, ' nu=', self.nu, ', c=', self.cc, ', a=', self.aa, ' lambda=', self.lamda)

	def get_PDF(self):
		self.K = abs(gamma(complex(self.nu, self.cc)))**2/(self.aa*np.sqrt(np.pi)*gamma(self.nu)*gamma(self.nu - 0.5))             
		self.W = self.K*np.exp(2*self.cc*np.arctan((self.x - self.lamda)/self.aa))/(1 + ((self.x - self.lamda)/self.aa)**2)**self.nu
		return self.W


	def get_pulse(self, t):
		if self.sinusoid: 
			return self.E0*(1 + 0.1*np.cos(np.pi*t/0.2e-6))
		else:
			return self.E0*(np.exp(-t/0.2e-6))



	def calculate_p_bar(self):
		p_bar_at_t = np.sum(self.x*self.W)*(self.x[1]-self.x[0])
		return p_bar_at_t

	def get_u(self, tnow):
		self.p_bar = self.calculate_p_bar()
		self.u = self.sigma_e*self.get_pulse(tnow) + self.phi*self.p_bar
		return self.u


	def plot(self):
		plt.figure(figsize=(7,7))
		plt.plot(-self.x, self.W, color='k', linewidth=2)
		plt.xlim([self.xmin, self.xmax])
		plt.ylabel(r'$\rm W_s(p_z)$', fontsize=25)
		plt.xlabel(r'$\rm -p_z\ [A/mm^2]$', fontsize=25)
		plt.legend(fontsize=20, loc=2, frameon=False)
		plt.tight_layout()
		plt.show()
		pass

	def update_params(self, t, mu, s2):
		self.u_store.append(self.get_u(t))
		self.lamda = -self.epsilon*self.alpha_p*self.get_u(t)/self.gamma_p
		self.aa    = self.alpha_p*self.get_u(t)*np.sqrt(1- self.epsilon**2)/self.gamma_p
		self.nu    = (self.aa**2 + (mu - self.lamda)**2 + 3*s2)/(2*s2)
		self.cc    = (mu - self.lamda)*(self.aa**2 + (mu - self.lamda)**2 + s2)/(2*self.aa*s2)
		self.get_PDF()
		self.print_params(t)


	def dmu_sigma2(self, t, y):
		self.update_params(t, y[0], y[1])
		u       = self.get_u(t)
		dmu     = -self.gamma_b*y[0] - self.alpha_b*u + 0.5*self.gamma_p**2*y[0] + 0.5*self.epsilon*self.alpha_p*self.gamma_p*u
		dsigma2 = -2*(self.gamma_b - self.gamma_p**2)*y[1] + self.gamma_p**2*y[0]**2 + 2*self.epsilon*self.gamma_p*self.alpha_p*u*y[0] + self.alpha_p**2*u**2
		return [dmu, dsigma2]

	def Solve(self):
		y0 = [self.mu_ini, self.sigma2_ini]
		self.sol = solve_ivp(self.dmu_sigma2, [0, self.tfinal], y0, method='LSODA')
		self.times = self.sol.t
		self.mus     = self.sol.y[0]
		self.sigma2s = self.sol.y[1]
		plt.figure(figsize=(7,7))
		plt.plot(1e6*self.times, self.mus/np.max(abs(self.mus)), 'g', label=r'$\rm \mu/\vert \mu \vert_{\max} $')
		plt.plot(1e6*self.times, self.sigma2s/np.max(self.sigma2s), 'r', label=r'$\rm \sigma^2/\sigma^2_{\max}$')
		plt.xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		plt.legend(fontsize=15, frameon=False)
		plt.show()
		plt.ion()
		plt.figure(figsize=(7,7))
		counter = 0
		for tt, mu, s2 in zip(self.times, self.mus, self.sigma2s):
			self.update_params(tt, mu, s2)
			self.lam_store.append(self.lamda)
			self.aa_store.append(self.aa)
			self.nu_store.append(self.nu)
			self.cc_store.append(self.cc)
			plt.plot(-self.x, self.W, color='k', linewidth=2)
			plt.xlim([self.xmin, self.xmax])
			plt.ylim([0, 5])
			plt.ylabel(r'$\rm W_s(p_z)$', fontsize=25)
			plt.xlabel(r'$\rm -p_z\ [A/mm^2]$', fontsize=25)
			plt.legend(fontsize=20, loc=2, frameon=False)
			plt.tight_layout()
			plt.draw()
			#plt.savefig("./movie/snap_"+str(counter).zfill(4)+".png")
			plt.pause(.001)
			plt.clf()
			counter += 1
		plt.close()
		plt.figure(figsize=(7,7))
		plt.plot(1e6*self.times, self.aa_store, label=r'$\rm a$')
		plt.plot(1e6*self.times, self.nu_store, label=r'$\rm \nu$')
		plt.plot(1e6*self.times, self.cc_store, label=r'$\rm c$')
		plt.plot(1e6*self.times, self.lam_store, label=r'$\rm \lambda$')
		plt.ylim([-1,10])
		plt.xlabel(r'$\rm time\ [\mu s]$', fontsize=25)
		plt.legend(frameon=False, fontsize=20)
		plt.tight_layout()
		plt.show()


SEP = CAEP(0.8, -3, 3)
pdb.set_trace()



