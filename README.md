# CAEP
* Source code for,

"A fractional stochastic theory for interfacial polarization of cell aggregates"
by Pouria Mistani, Samira Pakravan and Frederic Gibou

----------------------------------------------------------------------------------

* Dependencies: h5py, astroML, scipy, numpy, matplotlib, LaTex

----------------------------------------------------------------------------------

* Files: 

$ python CAEP_PDF_evolve.py
This is the main file with the actual solvers for the reduced order Fokker-Planck equations. 
There are 7 different test cases with different electric pulse profiles:

test_num = 0: a constant indefinite pulse

test_num = 1: a sinusoidal pulse with given frequency

test_num = 2: an exponential decay pulse

test_num = 3: a step pulse for 1 microseconds, then 1 microseconds of zero pulse.

test_num = 4: The case considered in the paper and compared with direct numerical simulations.

test_num = 5: The Gaussian pulse considered in the paper. For this test there are four different "cases" with different sets of matrix and cytoplasm conductivities, membrane conductance, volume fraction and duration of integration.

test_num = 6: a smoothed step pulse, i.e. using a sigmoid function instead of a discontinuous step pulse.

----------------------------------------------------------------------------------


$ python FourierSpace.py
This command will vary different parameters and makes the Cole-Cole and Bode plots reported in the paper.