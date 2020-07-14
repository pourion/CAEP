import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=15, usetex=True)

SL = 1.9
R1 = 7.0e-6
Cm = 0.01


sigma_e = 0.6
nu_list=[1.3/sigma_e]

sigma_e = 1.3
nu_list=[0.6/sigma_e]

phi = np.linspace(0, 0.9, 200)
freq  = np.logspace(2, 8, 400)    
PHI, FREQ = np.meshgrid(phi, freq)

for nu in nu_list:
	print(nu)
	sigma_c = sigma_e*nu
	SIGMA_t = 2*sigma_e + sigma_c + PHI*(sigma_e - sigma_c)

	g0 = SL*R1/sigma_c
	eta = 1 + g0*(1 + nu*(1-PHI)/(2 + PHI))
	tau = SIGMA_t*R1*Cm/(sigma_e*sigma_c*(2+PHI))
	alpha_p = -3.0/((2+PHI)*(eta + complex(0,1)*2*np.pi*FREQ*tau))
        alpha_s = -3.0*(sigma_e - sigma_c)*(eta-1+complex(0,1)*2*np.pi*FREQ*tau)/( (eta+complex(0,1)*2*np.pi*FREQ*tau) * SIGMA_t)
	k = ( (2+3*PHI*sigma_e) + sigma_c )/SIGMA_t - 3*PHI**2*sigma_c/( (2+PHI)*SIGMA_t*(eta+complex(0,1)*2*np.pi*FREQ*tau) )

        chi_p = PHI*alpha_p / k
        chi_s = PHI*alpha_s / k

	sigma_bar = 1 + chi_s / (chi_p + chi_s)
	Ee_per_Eext = (1 + chi_p + chi_s*sigma_e/(sigma_e - sigma_c))/(1-PHI)

	H = 1.0e-3
        A = 1.0e-6
        Ze = H/(sigma_e*A)
        Z = Ze/(1 + chi_p + chi_s)
        Z *= 1e-3


	fig = plt.figure(figsize=(7,7))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(np.log10(FREQ), PHI, np.abs(Ee_per_Eext), cmap=cm.PiYG_r, linewidth=0.5, antialiased=True)
        ax.set_xlabel(r'$\rm \log_{10}(frequency/Hz)$', fontsize=20, labelpad=20)
        ax.set_ylabel(r'$\rm \phi$', fontsize=20, labelpad=20)
	ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$\rm\frac{\vert\tilde{\mathbf{E}}_e\vert}{\vert \tilde{\mathbf{E}}_{ext}\vert}$', fontsize=20, labelpad=20, rotation=0)
        if nu<1:
                ax.view_init(elev=20., azim=50)
        else:
                ax.view_init(elev=20., azim=-115)
        #ax.set_title(r'$\rm \nu\ :\ $'+str(nu), pad=20 )
        plt.tight_layout()
        plt.show()



	fig = plt.figure(figsize=(7,7))
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(np.log10(FREQ), PHI, np.real(sigma_bar), cmap=cm.PiYG_r, linewidth=0.5, antialiased=True)
	ax.set_xlabel(r'$\rm \log_{10}(frequency/Hz)$', fontsize=20, labelpad=20)
	ax.set_ylabel(r'$\rm \phi$', fontsize=20, labelpad=20)
	ax.set_zlabel(r'$\rm Re(\bar{\sigma}/\sigma_e)$', fontsize=20, labelpad=20, rotation=270)
	if nu<1:
		ax.view_init(elev=20., azim=50)
	else:
		ax.view_init(elev=20., azim=-115)
	#ax.set_title(r'$\rm \nu\ :\ $'+str(nu), pad=20 )
	plt.tight_layout()
	plt.show()

	fig = plt.figure(figsize=(7,7))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(np.log10(FREQ), PHI, np.imag(sigma_bar), cmap=cm.PiYG_r, linewidth=0.5, antialiased=True)
        ax.set_xlabel(r'$\rm \log_{10}(frequency/Hz)$', fontsize=20, labelpad=20)
        ax.set_ylabel(r'$\rm \phi$', fontsize=20, labelpad=20)
        ax.set_zlabel(r'$\rm Im(\bar{\sigma}/\sigma_e)$', fontsize=20, labelpad=20, rotation=270)
	ax.tick_params(axis='z', pad=10)
	if nu<1:
        	ax.view_init(elev=20., azim=75)
	else:
                ax.view_init(elev=20., azim=-105)
        #ax.set_title(r'$\rm \nu\ :\ $'+str(nu), pad=20 )
        plt.tight_layout()
        plt.show()


	fig = plt.figure(figsize=(7,7))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(np.log10(FREQ), PHI, np.abs(sigma_bar), cmap=cm.PiYG_r, linewidth=0.5, antialiased=True)
        ax.set_xlabel(r'$\rm \log_{10}(frequency/Hz)$', fontsize=20, labelpad=20)
        ax.set_ylabel(r'$\rm \phi$', fontsize=20, labelpad=20)
        ax.set_zlabel(r'$\rm \vert \bar{\sigma}/\sigma_e\vert$', fontsize=20, labelpad=20, rotation=270)
        ax.tick_params(axis='z', pad=10)
        if nu<1:
                ax.view_init(elev=20., azim=75)
        else:
                ax.view_init(elev=20., azim=-105)
        #ax.set_title(r'$\rm \nu\ :\ $'+str(nu), pad=20 )
        plt.tight_layout()
        plt.show()



	fig = plt.figure(figsize=(7,7))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(np.log10(FREQ), PHI, np.abs(Z), cmap=cm.PiYG_r, linewidth=0.5, antialiased=True)
        ax.set_xlabel(r'$\rm \log_{10}(frequency/Hz)$', fontsize=20, labelpad=20)
        ax.set_ylabel(r'$\rm \phi$', fontsize=20, labelpad=20)
        ax.set_zlabel(r'$\rm \vert Z \vert (k\Omega)$', fontsize=20, labelpad=30, rotation=270)
	ax.tick_params(axis='y', pad=10)
        ax.tick_params(axis='z', pad=15)
	if nu<1:        
		ax.view_init(elev=20., azim=-150)
	else:
                ax.view_init(elev=20., azim=70)
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(7,7))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(np.log10(FREQ), PHI, np.angle(Z), cmap=cm.PiYG_r, linewidth=0.5, antialiased=True)
        ax.set_xlabel(r'$\rm \log_{10}(frequency/Hz)$', fontsize=20, labelpad=20)
        ax.set_ylabel(r'$\rm \phi$', fontsize=20, labelpad=20)
        ax.set_zlabel(r'$\rm \angle Z\ (rad)$', fontsize=20, labelpad=30, rotation=270)
        ax.tick_params(axis='y', pad=10)
	ax.tick_params(axis='z', pad=15)
	if nu<1:
	        ax.view_init(elev=20., azim=-95)
	else:
                ax.view_init(elev=20., azim=80)
        #ax.set_title(r'$\rm \nu\ :\ $'+str(nu), pad=20 )
        plt.tight_layout()
        plt.show()

