import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as st
from glob import glob
from csv import reader
from numpy import (random, square, array, concatenate, diff, arange,mean, absolute, empty, ones, cos, pi, multiply, isnan, invert, nan, 
                   exp, nan_to_num)
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import time
import datetime

print("Started program for SIR non Coupled Model")

start_time = time.time()

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

parameterErr = ["austriaAvg.csv","belarusAvg.csv", "belgiumAvg.csv", "bosniaAvg.csv","bulgariaAvg.csv","croatiaAvg.csv","denmarkAvg.csv","finlandAvg.csv","franceAvg.csv","germanyAvg.csv","greeceAvg.csv","irelandAvg.csv","netherlandsAvg.csv","norwayAvg.csv","portugalAvg.csv","romaniaAvg.csv","serbiaAvg.csv","ukAvg.csv","switzerlandAvg.csv"]
Threshold_Sir = [1e-06, 1e-04, 1e-06, 1e-07, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-07, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-05, 1e-04, 1e-06]
parameterPop = [9006398, 9449323, 11589623, 3280819, 6948445, 4105267, 5792202, 5540720, 65411076, 83783942, 10423054, 4937786, 83783942, 5421241, 10196709, 19237691, 8737371, 67886011, 8654622]
Cutoff = [200, 230, 220, 200, 200, 220,230, 200, 200, 200, 200, 200,200, 200,200, 200, 200, 250, 200]
Threshold_Sir = [1e-07, 1e-05, 1e-07, 1e-05, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-07, 1e-05, 1e-05, 1e-07]
x0_interval = [(0.0,0.20), (0.0,0.20), (0.0,0.20),(0.0,0.10),(0.0,0.25),(0.0,0.20),(0.0,0.10),(0.0,0.10),(0.0,0.10), (0.0,0.10),(0.0,0.10),(0.0,0.20),(0.0,0.10),(0.0,0.20),(0.0,0.10),(0.0,0.20),(0.8,1.0),(0.0,0.10),(0.0,0.10)]
Slope =  [0.06934346036647857, 0.0016310118021636958, 0.004221666441099336, 0.000726405697265338, 0.003540765699666154, 0.0019568859502787903, 0.014734266910518084, 0.0023959384701493355, 0.009448252903740085,
		   0.002096359221057019, 0.3094450795034076, 0.0031054538629794663, 0.0017298099948708308, 0.0009194808979929659, 0.04407759182469781, 0.003706543013345508, 0.002104872102345562, 0.01981831557400708, 0.005075305571492535]
Intercept =  [1.7840953632412493, 1.094006801941228, 1.7820776059392005, 0.47989213651816853, 0.42571774493546743, 0.6316209291557343, 6.396996694656187, 1.3743892490491856, 1.8291814440832699, 1.0767428548978368, 0.3094450795034076,
			   1.7362255483343263, 1.7149874636391265, 2.1909450765223557, 0.5566047256497786, 0.1985179903894998, 0.8173794007247331, 2.0070370448368875, 1.3438610557768937]





cutoff = Cutoff[idx]
slope = Slope[idx]
intercept = Intercept[idx]
Countries = '/mnt/hpc/work/s2frimpo/sirmodel/Countries/'
filename = Countries + parameterErr[idx][:-4] +"7ma.csv"
filename_osi = Countries + parameterErr[idx][:-4]+ "osi7ma.csv"
pop = parameterPop[idx]

logging.basicConfig(filename = "example_sir.log", encoding ='utf-8', level = logging.INFO)

logging.debug("This is debug message")

def pdf(x, mean, sdev): # unscaled probability density function (1/sqrt(2pi) term omitted because weights are rescaled anyway)
	
	e1 = 1/sdev
	e2 = e1*(x-mean)
	return e1*exp(-0.5*e2*e2)



def particleProb(oldParticle, newParticle): # find unscaled probability of a perturbed particle given the original
	
	mask = isnan(params)
	# ranges and bounds define permissible random param values
	ranges = array([r_alpha[1] - r_alpha[0], r_beta[1] - r_beta[0], r_b[1] - r_b[0], r_gamma[1] - r_gamma[0], r_c[1] - r_c[0], r_eta_0[1] - r_eta_0[0], r_epsilon[1] - r_epsilon[0], r_eta_1[1] - r_eta_1[0], r_kappa[1] - r_kappa[0], r_lamb[1] - r_lamb[0], r_sigma[1] - r_sigma[0], r_phi[1] - r_phi[0], r_E0[1] - r_E0[0], r_I0[1] - r_I0[0], r_x0[1] - r_x0[0], r_Ii0[1] - r_Ii0[0]])[mask]
	prob = 1 # initialize probability
	for i in range(len(oldParticle)):
		prob = prob*pdf(newParticle[i], oldParticle[i], sdevFrac*ranges[i]) # probability of a particle is product of probabilities of all its params
	return prob


def error(sim,x0,x1):
    err = mean(square(data - sim[int(t_data[0]):]))
    return err



def error_x(X):
    err = mean(square(stringency_data - X[int(t_data[0]):]))
    return err

def eta_func(t, eta_0, eta_1):
	eta_value = eta_0*(1-exp(-eta_1*t))
	return eta_value

def eta_linear_function(t, slope, intercept):
    return slope*t + intercept



def ode(params, variant, t_end): # returns simulation data for given params, ICs, and duration
	
	alpha   = params[0]
	beta    = params[1] # unpack the values in params into local variables
	b       = params[2] # local variables are faster to reference than array elements
	gamma   = params[3]
	c       = params[4]
	eta_0   = params[5]
	epsilon = params[6]
	eta_1     = params[7]
	kappa   = params[8]
	lamb    = params[9]
	sigma   = params[10]
	phi     = params[11]
	E0      = params[12]
	I0      = params[13]
	x0      = params[14]
	Ii0     = params[15]
	ic = array([1 - I0, I0, 0, Ii0])
	etaInv = 1/eta_0
	if variant == 3:
		def fun(t,y):
			e1 = beta*(1+b*cos((t-phi)*freq))*(1-0.2*epsilon)*y[0]*y[1]
			e2 = gamma*y[1]
			return [-e1, e1 - e2, e2, e1 ]
	# use RK45 method, with tolerances chosen to eliminate errors (wider tolerances are faster to run but may diverge for certain params)
	# the timeseries is state variable number 6 = y[5] which corresponds to Ii, and it's padded with a 0 for t=0 days at the beginning
	sol = solve_ivp(fun, (0.0, t_end+1), ic, method="RK45", t_eval=arange(0, t_end+1, 1), rtol=1e-9, atol=1e-9)
	T = sol.t
	S = sol.y[0]
	I = sol.y[1]
	R = sol.y[2]
	C = sol.y[3]
	return etaInv*concatenate([[0],diff(C)])




def getData(filename): # gets time and daily incidence data from .csv and stores into numpy arrays

	data = empty(730) # oversize matrix for data
	t_data = empty(730) # oversize matrix for data times
	files = glob(filename)
	for file in files:
		with open(file) as f:
			f = reader(f, delimiter=',')
			rows = 0
			for row in f:
				t_data[rows] = row[0]
				data[rows] = row[1]
				rows += 1
	return t_data[0:rows], data[0:rows]

def getData1(filename):
	data = pd.read_csv(filename)
	data_reported_cases = data[data.columns[-1]]
	t_data = np.arange(1,len(data_reported_cases)+1,1)
	return t_data, data_reported_cases



def rejection(N,filename, errAcpt, errAcptx): # rejection abc with uniform priors
	logging.info("The start of rejection method.")

	ti = time.time() # start time
	successes = 0 # successful tries counter
	fails = 0 # failed tries counter
	bounds = [r_alpha, r_beta, r_b, r_gamma, r_c, r_eta_0, r_epsilon, r_eta_1, r_kappa, r_lamb, r_sigma, r_phi, r_E0, r_I0, r_x0, r_Ii0]
	consts = array([bound[0] for bound in bounds]) # consts and coeffs map the [0,1) random.rand range to the param ranges
	coeffs = array([bound[1] for bound in bounds]) - consts
	length = len(params)
	nans = empty(length)
	nans[:] = nan
	trials = [] # list of trials
	while successes < N:
		particle = multiply(random.rand(length), coeffs) + consts # generate the random parameters (uniform dist [0,1])
		mask = isnan(params)
		particle = multiply(1*mask, particle) + nan_to_num(params)
		sim = ode(particle, variant, t_data[-1]) # simulate the random parameters
		negative_values = [neg_val for neg_val in sim if neg_val < 0]
		if len(negative_values) == 0 and len(sim) == t_data[-1] + 1 and error(sim,x0,x1)< errAcpt:
			trials.append((error(sim,x0,x1), particle[mask])) # store in tuple : error, fitted params
			successes += 1
		else: # indicates simulation diverged before t_end
			fails += 1
	print ("Acceptance Rate: ", successes/(fails+successes), fails, "fails and", successes, "successes in", str(datetime.timedelta(seconds=(time.time()-ti))))
	sortedTrials = sorted(trials, key=lambda x:x[0]) # sort particles
	print("best error:", (sortedTrials[0])[0])
	logging.info("Completed rejection method")
	return sortedTrials



def smcSample(N, nParams, nSamples, samples, weights,filename, errAcpt, errAcptx): # takes samples from previous population and generates new population

	ti = time.time()
	mask = isnan(params)
	# ranges and bounds define permissible random param values
	ranges = array([r_alpha[1] - r_alpha[0], r_beta[1] - r_beta[0], r_b[1] - r_b[0], r_gamma[1] - r_gamma[0], r_c[1] - r_c[0], r_eta_0[1] - r_eta_0[0], r_epsilon[1] - r_epsilon[0], r_eta_1[1] - r_eta_1[0], r_kappa[1] - r_kappa[0], r_lamb[1] - r_lamb[0], r_sigma[1] - r_sigma[0], r_phi[1] - r_phi[0], r_E0[1] - r_E0[0], r_I0[1] - r_I0[0], r_x0[1] - r_x0[0], r_Ii0[1] - r_Ii0[0]])[mask]
	bounds = array([r_alpha, r_beta, r_b, r_gamma, r_c, r_eta_0, r_epsilon, r_eta_1, r_kappa, r_lamb, r_sigma, r_phi, r_E0, r_I0, r_x0, r_Ii0])[mask]
	np.set_printoptions(precision=0)
	print("weights:", array(weights))
	successes = 0
	fails = 0
	trials = [] # list of trials
	while successes < N:
		choice = random.choice([*range(nSamples)], p=weights) # randomly choose a particle from samples with weights
		chosen = (samples[choice])[1]
		perturbed = empty(nParams) # perturb chosen particle
		perturbed[:] = nan
		for param in range(nParams):
			param_bounds = bounds[param]
			while not (param_bounds[0] <= perturbed[param] <= param_bounds[1]): # if perturbed param out of bounds, try again
				perturbed[param] = random.normal(chosen[param], sdevFrac*ranges[param]) # random gaussian walk centred around original value with sdev of sdevFrac times the param range
		particle = np.copy(params) # reconstruct params array
		counter = 0
		for i in range(len(particle)):
			if isnan(particle[i]):
				particle[i] = perturbed[counter]
				counter += 1
		sim = ode(particle, variant, t_data[-1]) # simulate the random parameters
		negative_values = [neg_val for neg_val in sim if neg_val < 0]
		if len(negative_values) == 0 and len(sim) == t_data[-1] + 1 and error(sim,x0,x1)< errAcpt:
			trials.append((error(sim,x0,x1), particle[mask])) # store in tuple : error, fitted params
			successes += 1
		else:
			fails += 1
	print ("Acceptance Rate: ", successes/(fails+successes), fails, "fails and", successes, "successes in", str(datetime.timedelta(seconds=(time.time()-ti))))
	sortedTrials = sorted(trials, key=lambda x:x[0]) # sort particles
	print("best error:", (sortedTrials[0])[0])
	return sortedTrials



def smc(N,filename, errAcpt, errAcptx): # generates list of particles from successive iterations of smcSample()
	logging.info("The start of SMC method.")

	bestErrors = []
	Threshold = []
	nSamples = int(N*sampleFrac) # number of samples : take top sampleFrac of each population
	pop0 = rejection(N,filename, errAcpt, errAcptx) # first do rejection to generate initial samples
	bestErrors.append((pop0[0])[0])
	trials = pop0 #initialize list of all particles
	samples = pop0[0:nSamples]
	nParams = len((samples[0])[1]) # number of params to be fitted
	weights = (ones(nSamples)/nSamples).tolist() # initialize weights to be all equal
	pOld = ones(nSamples) # initialize theta_{t-1} probabilities to be all equal 
	for pop in range(nPop):
		error_sample = [samples[i][0] for i in range(nSamples)] #collects the mean square error from previous population
		new_errAcpt = np.max(error_sample) #we select the new error threshold as the maximum value of the 
		if new_errAcpt < errAcpt:
			errAcpt = new_errAcpt
		else:
			errAcpt = errAcpt
		Threshold.append(errAcpt)
		popTrials = smcSample(N, nParams, nSamples, samples, weights,filename, errAcpt, errAcptx) # generate population from previous samples
		bestErrors.append((popTrials[0])[0])
		trials += popTrials # append newly generated particles
		newSamples = popTrials[0:nSamples]
		newWeights = []

		pNew  = empty(nSamples) # initialize theta_{t} probabilities
		for i in range(nSamples):
			w = 0
			pNew[i] = particleProb((samples[i])[1], (newSamples[i])[1]) # set the new particle probabilities
			for j in range(nSamples):
				#w += weights[j]*pOld[j]*pNew[i]
				w += weights[j]*pNew[i]
			w = 1/w
			newWeights.append(w) # set the new weights

		samples = newSamples # update samples
		weights = newWeights # update weights
		weights = (weights/np.sum(weights)).tolist()
		pOld = pNew # update particle probabilities

	print("Threshold ", Threshold)
	sorted_particles = trials
	best = sorted_particles
	parameters = [best[i][1] for i in range(len(sorted_particles))]
	Error = [best[i][0] for i in range(len(sorted_particles))]
	parameterset = pd.DataFrame(parameters,columns=[ 'beta', 'b', 'gamma',"eta_0", 'phi', 'I0', "Ii0"])
	parameterset["Error"] = Error
	parameterset.to_csv(parameterErr[idx][:-4] + "_" + str(n) + "_" + "sir_trial.csv",index=False)
	logging.info("The end of SMC method for all population.")  
	return None


def plotTrials(trials, variant, t_end): # plot list of trials on the same graph
	Simulation = []    
	plt.figure() # start a new plot    
	for trial in trials:
		particle = np.copy(params)
		counter = 0
		fitted = trial[1]
		for i in range(len(particle)):
			if isnan(particle[i]):
				particle[i] = fitted[counter]
				counter += 1
		trial_data = ode(particle, variant, t_end)
		Simulation.append(trial_data)
		t = arange(0, len(trial_data), 1)         
		plt.plot(t, trial_data, label='{:g}'.format(float('{:.2g}'.format(trial[0]))))
	plt.scatter(t_data, data, s=5, c='lime') # plot the real data
	plt.scatter(t_data_full[cutoff:], data_full[cutoff:], s=5, c='magenta') # plot the real data
	plt.title(filename + "\t" + str(errAcpt))
	plt.legend(loc = "upper left") 
	return None



def plotBestErrors(bestErrors):
	plt.figure() # start a new plot
	pops = arange(0, nPop+1, 1)
	plt.plot(pops, bestErrors)
	plt.yscale('log')
	plt.title('best error vs population number')
	return None


# define fixed (known) params, params to be fitted are nan, params that are N/A are assigned 0 because anything nonzero slows it down unnecessarily
# gamma = 0.2, epsilon = 0.63, sigma k= 0.4
# params      =       [   alpha,    beta,       b,   gamma,       c,   eta_0, epsilon,     eta_1,   kappa,  lambda,   sigma,     phi,      E0,      I0,      x0,     Ii0]
params_v0     = array([     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan])
params_v1     = array([       0,     nan,     nan,     nan,     nan,       0,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan])
params_v2     = array([       0,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan])
params_v3     = array([       0,     nan,     nan,     nan,     0,     nan,     1.0,     0,     0,     0,     0,     nan,     0,     nan,     0.2,     nan])
params_v4     = array([       0,     nan,     nan,     nan,     nan,       0,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan])
params_v5     = array([       0,     nan,     nan,     nan,       0,       0,     nan,     nan,       0,       0,     nan,     nan,     nan,     nan,     nan,     nan])

variants      = [params_v0, params_v1, params_v2, params_v3, params_v4, params_v5]

plt.figure(figsize=(7,7))
variant = 3
params = variants[variant]
params.flags.writeable = False # throw an error if a bug causes params to be modified


####################################################################################################
#												MAIN											   #
####################################################################################################
freq = 2*pi/365

# other parameters for smc
sampleFrac = 0.7 # proportion of each population that makes it into the sample
sdevFrac = 0.15 # paramRange*sdevFrac = sdev for gaussian random walk
nPop = 20 # number of smc populations
popSize = 1000 # number of particles in a population
printEvery = 10 # print progress every printEvery simulation runs
bestN = 10 # plot the bestN best particles at the end


#------ ORIGINAL RANGE -------

r_alpha   = (      0,      2)
r_beta    = (   0.1,    0.9) 		#(0.3, 0.9)
r_b       = (      0.0,    1.0)
r_gamma   = (   0.1,   0.5) 		#(0.1, 0.5)
r_c       = (      0,  0.01) 		#(0, 0.1)
r_eta_0   = (      1,  10)
r_epsilon = (    0.1,    0.8) 		#(0.3, 0.8)
r_eta_1     = (      1,   10)
r_kappa   = (      100,  5000) 		#(100, 5000)
r_lamb    = (      0.0,   0.2) 
r_sigma   = (    0.2,    0.9) 		#(0.2, 0.9)
r_phi     = (    -135,    -45)


print("start at", time.strftime("%H:%M:%S", time.localtime()))
ti = time.time()
errAcpt = Threshold_Sir[idx]*10 
errAcptx = 0.09 
x0 = 66
x1 = 200
popInv = 1/pop
r_I0      = (     1*popInv,     2000*(1/pop ))
r_x0      = x0_interval[idx]
r_Ii0     = (     0*popInv,     1000*(1/pop ))
r_E0      = (     1*(1/pop),     500*(1/pop))

t_data_full, data_full = getData1(filename)
t_data_full_osi, data_full_osi = getData1(filename_osi)
t_data_full = np.arange(1,len(t_data_full)+1)
t_end = 400
stringency_data = data_full_osi[:cutoff]
t_data_osi = t_data_full_osi[:cutoff]
t_data = t_data_full[:cutoff]
data = data_full[:cutoff]

print("All function check out")


for h in range(10):
	n = h
	smc(popSize,filename, errAcpt, errAcptx)

print("completed!!")
#################################################################################################################################################################

end_time = time.time()

# Compute the elapsed time
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.6f} seconds")


