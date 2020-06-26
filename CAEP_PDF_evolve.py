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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class CAEP:
	def __init__(self, epsilon, xmin, xmax, testnum, tf, maxstep =2e-9, eval_points=1000, Nx=2000, phi=0.13):
		self.animate = False
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
		self.sigma_e = 1.3*self.scaling      #15                       # S/mm
		self.sigma_c = 0.6*self.scaling      #1                 # S/mm
		self.Cm      = 0.00983*self.scaling**2                      # F/mm^2
		self.SL      = 1.9*self.scaling**2                       # S/mm^2
		self.R       = 7.0e-6/self.scaling                       # mm
		self.phi     = phi

		self.sigma_t = 2*self.sigma_e + self.sigma_c + self.phi*(self.sigma_e - self.sigma_c)
		self.alpha_b = 3*self.sigma_e*self.sigma_c/(self.Cm*self.R*self.sigma_t)
		self.gamma_b = self.SL/self.Cm + ( self.sigma_e * self.sigma_c * (2 + self.phi))/(self.R*self.Cm*self.sigma_t) 

		# experiments
		self.nu_exp = 7.246803016058461
		self.cc_exp = 0.8884126275039926
		self.aa_exp = 0.1635772557705042
		self.lamda_exp = 0.8644983717186614
		self.gamma_p = np.sqrt(self.gamma_b/(self.nu_exp - 0.5))
		self.epsilon = 1.0/np.sqrt(1.0 + self.aa_exp**2/self.lamda_exp**2)
		self.alpha_p = self.alpha_b*self.gamma_p/(self.epsilon*self.gamma_b - self.cc_exp*self.gamma_p**2*np.sqrt(1 - self.epsilon**2))
		self.u_exp   = -self.lamda_exp*self.gamma_p/(self.epsilon*self.alpha_p)

		# stochastic parameters
		# self.epsilon = epsilon                          # correlation between alpha and gamma \in [-1, 1]
		# self.gamma_p = gamma_p #np.sqrt(gm_factor*self.gamma_b)  # noise in gamma       
		# self.alpha_p = alpha_p #np.sqrt(ap_factor*self.alpha_b)  #*self.alpha_b*(self.gamma_p/self.gamma_b)/self.epsilon  

		self.eta     = 1 + self.SL*self.sigma_t*self.R/((2 + self.phi)*self.sigma_e*self.sigma_c)
		self.tau     = self.Cm*self.sigma_t*self.R/((2 + self.phi)*self.sigma_e*self.sigma_c)
		self.omega   = 1.0e6
		# a = -3/((2+self.phi)*(self.eta + complex(0,1)*self.omega*self.tau))
		# b = -3*(self.sigma_e-self.sigma_c)*(self.eta-1+complex(0,1)*self.omega*self.tau)/((self.sigma_t)*(self.eta+complex(0,1)*self.omega*self.tau))
		# self.chi_p   = a*self.phi
		# self.chi_s   = (1 + a*self.phi)*b*self.phi
		# self.chi     = self.chi_p + self.chi_s
		


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

	def effective_cond(self, w=0):
		b = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*w*self.tau ) /(self.sigma_t*(self.eta + complex(0,1)*w*self.tau))
		return 1 + b*self.phi/(1.0 - self.phi*(1 + b*self.sigma_e/(self.sigma_e-self.sigma_c)))

	def get_pulse(self, t):
		if self.test==0:
			return self.E0
		elif self.test==1: 
			return self.E0*np.cos(np.pi*t/0.2e-6)
		elif self.test==2:
			return self.E0*(np.exp(-t/0.2e-6))
		elif self.test==3:
			return self.E0*np.sin(self.omega*t)
		elif self.test==4:
			if t<1e-6:
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
		omega = 0
		A = 1 + (self.phi*(1 - self.phi)/(2+nu + self.phi*(1-nu)))*(1-nu + 3*nu*self.eta/((2+self.phi)*(self.eta + omega*self.tau)))
		# B = -3*nu*omega*self.tau*self.phi*(1-self.phi)/((2+nu + self.phi*(1-nu))*(2+self.phi)*(self.eta**2 + omega**2*self.tau**2))
		E_e = E_ext/abs(A)
		self.u = self.sigma_e*E_e 
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
		dmu     = -self.gamma_b*y[0] - self.alpha_b*u + 0.5*self.gamma_p**2*y[0] + 0.5*self.epsilon*self.alpha_p*self.gamma_p*u
		dsigma2 = -2*(self.gamma_b - self.gamma_p**2)*y[1] + self.gamma_p**2*y[0]**2 + 2*self.epsilon*self.gamma_p*self.alpha_p*u*y[0] + self.alpha_p**2*u**2
		return [dmu, dsigma2]

	def jacob(self, t,y):
		return np.array([[-self.gamma_b + 0.5*self.gamma_p**2, 0], [2*self.gamma_p**2*y[0]+2*self.epsilon*self.gamma_p*self.alpha_p*self.get_u(t,y[0]), -4*(self.gamma_b-self.gamma_p**2)*np.sqrt(y[1]) ] ])

	def Solve(self):
		self.t_evaluation = np.linspace(0, self.tfinal, self.eval_point)
		y0 = [self.mu_ini, self.sigma2_ini]
		self.sol = solve_ivp(self.dmu_sigma2, [0, self.tfinal], y0, method='LSODA', rtol=1e-15, jac=self.jacob, max_step=self.maxstep, t_eval=self.t_evaluation) 
		self.times   = self.sol.t
		self.mus     = self.sol.y[0]
		self.sigma2s = self.sol.y[1]
		self.Es = np.array([self.get_pulse(t) for t in self.times])
		
		if self.test==4:
			data_sim = np.loadtxt("data_time_mean_variance_ps.dat")
			fig, ax = plt.subplots(1, 2, figsize=(15,7))
			ax[0].plot(1e6*self.times, -self.mus, 'r', linestyle='-', linewidth=2, label=r'$\rm p_z\ model$')
			ax[0].plot(data_sim[:,0], data_sim[:,1], color='g', linestyle=':', linewidth=1, label=r'$\rm p_x\ simulation$')
			ax[0].plot(data_sim[:,0], data_sim[:,2], color='b', linestyle='-.', linewidth=1, label=r'$\rm p_y\ simulation$')
			ax[0].scatter(data_sim[:,0], data_sim[:,3], color='k', marker='o', s=10, label=r'$\rm p_z\ simulation$')
			ax[1].plot(1e6*self.times, self.sigma2s, 'r', linestyle='-', linewidth=2, label=r'$\rm p_z\ model$')
			ax[1].plot(data_sim[:,0], data_sim[:,4], color='g', linestyle=':', linewidth=1, label=r'$\rm p_x\ simulation$')
			ax[1].plot(data_sim[:,0], data_sim[:,5], color='b', linestyle='-.', linewidth=1, label=r'$\rm p_y\ simulation$')
			ax[1].scatter(data_sim[:,0], data_sim[:,6], color='k', marker='o', s=10, label=r'$\rm p_z\ simulation$')
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

	def fourier_analysis(self):
		N = len(self.times)
		T = self.times[1] - self.times[0]
		yf = scipy.fftpack.fft(self.mus)

		self.Pf = (2.0/N) * yf[:N//2]
		self.yf = abs(self.Pf) 
		self.xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
		
		uf = scipy.fftpack.fft(self.u_store)
		Ef = scipy.fftpack.fft(self.Es)
		self.Eextf = (2.0/N) * Ef[:N//2]
		self.uf = (2.0/N) * np.abs(uf[:N//2])
		self.Ef = (2.0/N) * np.abs(Ef[:N//2])

		# self.sigma_eff = self.effective_cond(2*np.pi*self.xf)
		# self.Z = 1.0/self.sigma_eff
		# self.Z = self.Eextf/(self.sigma_e*self.Eextf + self.phi*(1 + self.chi_s/self.chi_p)*self.Pf )
		self.a = -3/(2+self.phi)/(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau )
		self.b = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*2*np.pi*self.xf*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau))
		self.chi_s = self.b*self.phi/(1 - self.phi*(1 + self.a + self.b*self.sigma_e/(self.sigma_e - self.sigma_c)))
		self.chi_p = self.a*self.phi/(1 - self.phi*(1 + self.a + self.b*self.sigma_e/(self.sigma_e - self.sigma_c)))
		
		self.Znormal = (self.sigma_e*self.Eextf + self.phi*self.Pf)/(self.sigma_e*self.Eextf + self.phi*self.Pf*(1+self.chi_s/self.chi_p)) 
		self.permittivity = 1.0/(complex(0,1)*2*np.pi*self.xf*self.Znormal/self.sigma_e)


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
		plt.ylabel(r'$\rm \vert \tilde{p}\vert \ [A/mm^2]$', fontsize=25)
		plt.yscale('log')
		plt.xscale('log')
		plt.tight_layout()
		plt.show()


		



		fig, axes = plt.subplots(2, figsize=(7,7))
		axes[0].plot(self.xf, np.real(self.permittivity), linewidth=2, color='k')
		axes[1].plot(self.xf, np.imag(self.permittivity), linewidth=2, color='k')
		axes[0].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[1].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
		axes[0].set_ylabel(r"$\rm \epsilon'/\epsilon_0$", fontsize=25)
		axes[1].set_ylabel(r"$\rm \epsilon''/\epsilon_0$", fontsize=25)
		# axes[0].set_yscale('log')
		axes[0].set_xscale('log')
		axes[1].set_xscale('log')
		#axes[0].set_ylim([0.82, 1.02])
		#axes[1].set_ylim([-0.1, 0])
		plt.tight_layout()
		plt.show()

		plt.figure(figsize=(7,7))
		plt.plot(np.real(self.permittivity), -np.imag(self.permittivity), linewidth=2, color='k')
		plt.xlabel(r"$\rm \epsilon'/\epsilon_0$", fontsize=25)
		plt.ylabel(r"$\rm -\epsilon''/\epsilon_0$", fontsize=25)
		# plt.xlim([60, 67])
		# plt.ylim([0, 7])
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
		# plt.ylim([0, Zmax-Zmin])
		plt.tight_layout()
		plt.show()


testnum    =  5           # 6: smoothed step function, 5: Gaussian, 4: sharp step pulse
tf         =  2e-5	      # [s]
eps        =  0.982       # eps = 1/(1 + a^2 /lamda^2)**0.5 = 0.982 
xmin       = -5.0
xmax       =  5.0
nx         =  2000
t_samples  =  200000
max_t_step =  1e-9

phi        =  0.37        #0.13*(4*np.pi/3.0)*(5e-4)**3/(1e-9)
sep        =  CAEP(eps, xmin, xmax, testnum, tf, max_t_step, t_samples, nx, phi)

sep.Solve()

# sep.evolution_pdf()

sep.fourier_analysis()

# pdb.set_trace()

