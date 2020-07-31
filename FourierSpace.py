from CAEP_PDF_evolve import CAEP
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=15, usetex=True)
import scipy.fftpack
from scipy import special

def get_colors(num, cmmp='coolwarm'):
		cmap = plt.cm.get_cmap(cmmp)
		cs = np.linspace(0, 1, num)
		colors = []
		for i in range(num):
			colors.append(cmap(cs[i]))
		return np.array(colors)

def FourierSpace_config_I(fraction=1):
	testnum     =  5
	xmin        = -5.0
	xmax        =  5.0
	nx          =  2000
	max_t_step  =  1e-9
	sigma_e     =  1.3
	sigma_c     =  0.6
	SL          =  1.9
	tf          =  2.0e-5         # [s]
	t_samples   =  1e5            # number of time samples recorded
	eps         =  0.982 
	NUM_CURVES  =  10
	phiiss      =  np.linspace(0.01, 0.8, NUM_CURVES)
	Zn_store    =  []
	counter     =  1
	for phi in phiiss:
		print("counter = ", counter, " out of = ", NUM_CURVES)
		counter+= 1
		self    = CAEP(eps, sigma_e, sigma_c, SL, xmin, xmax, testnum, tf, max_t_step, t_samples, nx, phi, False)
		self.solve_only(fraction)
		times   = self.times
		Es      = self.Es
		N       = len(times)
		T       = times[1] - times[0]
		mus     = self.mus
		yf      = scipy.fftpack.fft(mus)
		self.Pf = (2.0/N) * yf[:N//2]
		self.yf = abs(self.Pf) 
		self.xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
		Ef      = scipy.fftpack.fft(Es)
		self.Eextf   = (2.0/N) * Ef[:N//2]
		self.Ef      = (2.0/N) * np.abs(Ef[:N//2])
		self.alpha_p = -3/(2+self.phi)/(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau )
		self.alpha_s = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*2*np.pi*self.xf*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau))
		self.alpha_e = (2*self.sigma_e + (1 + self.phi*self.alpha_p)*self.sigma_c)/self.sigma_t
		self.k       = self.alpha_e - self.phi*self.alpha_p - self.phi*self.sigma_e*self.alpha_s/(self.sigma_e - self.sigma_c)
		self.chi_s   = self.alpha_s*self.phi / self.k
		self.chi_p   = self.alpha_p*self.phi / self.k
		area         = 1.0    # mm^2
		H            = 1.0    # mm
		Ze           = self.sigma_e*area/H
		Znormal      = (self.sigma_e*self.Eextf)/(self.sigma_e*self.Eextf + self.phi*self.Pf*(1-self.sigma_m/self.sigma_e) + self.phi*self.Pf*self.chi_s/self.chi_p ) 
		Zn_store.append(Znormal)
	Zn_store = np.array(Zn_store)
	return self.xf, Zn_store


def FourierSpace_config_II(fraction=1):
	NUM_CURVES  =  10
	testnum     =  5
	xmin        = -5.0
	xmax        =  5.0
	nx          =  2000
	max_t_step  =  1e-9
	sigma_e_s     =  np.linspace(0.5, 1.5, NUM_CURVES)
	sigma_c     =  1.0
	SL          =  1.9
	tf          =  2.0e-5         # [s]
	t_samples   =  1e5            # number of time samples recorded
	eps         =  0.982 
	phi         =  0.3
	Zn_store    =  []
	counter     =  1
	for sigma_e in sigma_e_s:
		print("counter = ", counter, " out of = ", NUM_CURVES)
		counter+= 1
		self    = CAEP(eps, sigma_e, sigma_c, SL, xmin, xmax, testnum, tf, max_t_step, t_samples, nx, phi, False)
		self.solve_only(fraction)
		times   = self.times
		Es      = self.Es
		N       = len(times)
		T       = times[1] - times[0]
		mus     = self.mus
		yf      = scipy.fftpack.fft(mus)
		self.Pf = (2.0/N) * yf[:N//2]
		self.yf = abs(self.Pf) 
		self.xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
		Ef      = scipy.fftpack.fft(Es)
		self.Eextf   = (2.0/N) * Ef[:N//2]
		self.Ef      = (2.0/N) * np.abs(Ef[:N//2])
		self.alpha_p = -3/(2+self.phi)/(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau )
		self.alpha_s = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*2*np.pi*self.xf*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau))
		self.alpha_e = (2*self.sigma_e + (1 + self.phi*self.alpha_p)*self.sigma_c)/self.sigma_t
		self.k       = self.alpha_e - self.phi*self.alpha_p - self.phi*self.sigma_e*self.alpha_s/(self.sigma_e - self.sigma_c)
		self.chi_s   = self.alpha_s*self.phi / self.k
		self.chi_p   = self.alpha_p*self.phi / self.k
		area         = 1.0    # mm^2
		H            = 1.0    # mm
		Ze           = self.sigma_e*area/H
		Znormal      = (self.sigma_e*self.Eextf)/(self.sigma_e*self.Eextf + self.phi*self.Pf*(1-self.sigma_m/self.sigma_e) + self.phi*self.Pf*self.chi_s/self.chi_p ) 
		Zn_store.append(Znormal)
	Zn_store = np.array(Zn_store)
	return self.xf, Zn_store

def FourierSpace_config_III(fraction=1):
	NUM_CURVES  =  10
	testnum     =  5
	xmin        = -5.0
	xmax        =  5.0
	nx          =  2000
	max_t_step  =  1e-9
	sigma_e     =  1.3
	sigma_c     =  0.6
	SL_s        =  np.logspace(np.log10(1.9), np.log10(1.9e5), NUM_CURVES)
	tf          =  2.0e-5         # [s]
	t_samples   =  1e5            # number of time samples recorded
	eps         =  0.982 
	phi         =  0.3
	Zn_store    =  []
	counter     =  1
	for SL in SL_s:
		print("counter = ", counter, " out of = ", NUM_CURVES)
		counter+= 1
		self    = CAEP(eps, sigma_e, sigma_c, SL, xmin, xmax, testnum, tf, max_t_step, t_samples, nx, phi, False)
		self.solve_only(fraction)
		times   = self.times
		Es      = self.Es
		N       = len(times)
		T       = times[1] - times[0]
		mus     = self.mus
		yf      = scipy.fftpack.fft(mus)
		self.Pf = (2.0/N) * yf[:N//2]
		self.yf = abs(self.Pf) 
		self.xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
		Ef      = scipy.fftpack.fft(Es)
		self.Eextf   = (2.0/N) * Ef[:N//2]
		self.Ef      = (2.0/N) * np.abs(Ef[:N//2])
		self.alpha_p = -3/(2+self.phi)/(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau )
		self.alpha_s = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*2*np.pi*self.xf*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau))
		self.alpha_e = (2*self.sigma_e + (1 + self.phi*self.alpha_p)*self.sigma_c)/self.sigma_t
		self.k       = self.alpha_e - self.phi*self.alpha_p - self.phi*self.sigma_e*self.alpha_s/(self.sigma_e - self.sigma_c)
		self.chi_s   = self.alpha_s*self.phi / self.k
		self.chi_p   = self.alpha_p*self.phi / self.k
		area         = 1.0    # mm^2
		H            = 1.0    # mm
		Ze           = self.sigma_e*area/H
		Znormal      = (self.sigma_e*self.Eextf)/(self.sigma_e*self.Eextf + self.phi*self.Pf*(1-self.sigma_m/self.sigma_e) + self.phi*self.Pf*self.chi_s/self.chi_p ) 
		Zn_store.append(Znormal)
	Zn_store = np.array(Zn_store)
	return self.xf, Zn_store


def FourierSpace_config_IV():
	NUM_CURVES  =  10
	testnum     =  5
	xmin        = -5.0
	xmax        =  5.0
	nx          =  2000
	max_t_step  =  1e-9
	sigma_e     =  1.3
	sigma_c     =  0.6
	SL          =  1.9
	tf          =  2.0e-5         # [s]
	t_samples   =  1e5            # number of time samples recorded
	eps         =  0.982 
	phi         =  0.3
	Zn_store    =  []
	counter     =  1
	fractions   = np.linspace(0.9, 1.1, NUM_CURVES)
	for fraction in fractions:
		print("counter = ", counter, " out of = ", NUM_CURVES)
		counter+= 1
		self    = CAEP(eps, sigma_e, sigma_c, SL, xmin, xmax, testnum, tf, max_t_step, t_samples, nx, phi, False)
		self.solve_only(fraction)
		times   = self.times
		Es      = self.Es
		N       = len(times)
		T       = times[1] - times[0]
		mus     = self.mus
		yf      = scipy.fftpack.fft(mus)
		self.Pf = (2.0/N) * yf[:N//2]
		self.yf = abs(self.Pf) 
		self.xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
		Ef      = scipy.fftpack.fft(Es)
		self.Eextf   = (2.0/N) * Ef[:N//2]
		self.Ef      = (2.0/N) * np.abs(Ef[:N//2])
		self.alpha_p = -3/(2+self.phi)/(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau )
		self.alpha_s = -3*(self.sigma_e-self.sigma_c)*(self.eta - 1 + complex(0,1)*2*np.pi*self.xf*self.tau )/(self.sigma_t*(self.eta + complex(0,1)*2*np.pi*self.xf*self.tau))
		self.alpha_e = (2*self.sigma_e + (1 + self.phi*self.alpha_p)*self.sigma_c)/self.sigma_t
		self.k       = self.alpha_e - self.phi*self.alpha_p - self.phi*self.sigma_e*self.alpha_s/(self.sigma_e - self.sigma_c)
		self.chi_s   = self.alpha_s*self.phi / self.k
		self.chi_p   = self.alpha_p*self.phi / self.k
		area         = 1.0    # mm^2
		H            = 1.0    # mm
		Ze           = self.sigma_e*area/H
		Znormal      = (self.sigma_e*self.Eextf)/(self.sigma_e*self.Eextf + self.phi*self.Pf*(1-self.sigma_m/self.sigma_e) + self.phi*self.Pf*self.chi_s/self.chi_p ) 
		Zn_store.append(Znormal)
	Zn_store = np.array(Zn_store)
	return self.xf, Zn_store



xf, Zs  = FourierSpace_config_IV()

colorss = get_colors(len(Zs))
Zmax    = np.max(np.real(Zs)) + 0.1
Zmin    = np.min(np.real(Zs)) - 0.1


fig, axes = plt.subplots(2, figsize=(7,7))
[ axes[0].plot(xf, np.abs(Zs[i]), linewidth=1, color=colorss[i]) for i in range(len(Zs)) ]
[ axes[1].plot(xf, np.angle(Zs[i]), linewidth=1, color=colorss[i]) for i in range(len(Zs)) ]
axes[0].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
axes[1].set_xlabel(r'$\rm frequency\ [1/s]$', fontsize=25)
axes[0].set_ylabel(r'$\rm \vert Z\vert/Z_e$', fontsize=25)
axes[1].set_ylabel(r'$\rm \angle Z\ [rad]$', fontsize=25)
axes[0].set_xscale('log')
axes[1].set_xscale('log')
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,7))
[ plt.plot(np.real(Zs[i]), -np.imag(Zs[i]), color=colorss[i]) for i in range(len(Zs)) ]
plt.xlabel(r'$\rm Re(Z) /Z_e$', fontsize=25)
plt.ylabel(r'$\rm -Im(Z)/Z_e$', fontsize=25)
plt.xlim([Zmin, Zmax])
plt.ylim([0, Zmax-Zmin])
plt.tight_layout()
plt.show()	


pdb.set_trace()


