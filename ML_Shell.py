import numpy as np
import time
import sys
import ML_Routines_2D
from astropy.io import ascii
from scipy.optimize import minimize
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

##########################################################################################################################
# Load the sample

stars = ascii.read('Gaia_DR3_Field_Giants_Feb_9_2023_Referee_Response', format = 'basic', guess = False) 

##########################################################################################################################
# Maximum likeklihood

def loglik(inputs, mag, pax, emag, epax, sinb,
           absrange   = [-5.0, -3.0],
           true_range = [-6.5, -1.5], 
           magrange   = [8.0, 11.7]):

    mbreak, alpha, gamma, hscale0 = inputs
    
#    alpha  = 0.3
#    gamma  = 0.3
    beta =  0.32107304 # 0.06694005
    dnorm0 = 1
    sig    = 0.004        
    
    #######################################################    
    # Likelihoods

    args = []
    for i in range(len(mag)):
        args.append((mag[i], pax[i], emag[i], epax[i], sinb[i], mbreak, alpha, beta, gamma, hscale0, dnorm0))

    with MPIPoolExecutor() as executor:
        liks = executor.starmap(ML_Routines_2D.likelihoods, args)

    likvec = []
    for likval in liks:
        likvec.append(likval)
    likvec = np.array(likvec)
    
    #######################################################
    # Normalizations

    args_norm = []
    for i in range(len(emag)):
        args_norm.append([emag[i], epax[i], sinb[i], mbreak, alpha, beta, gamma, hscale0, dnorm0])
    
    with MPIPoolExecutor() as executor:
        norms = executor.starmap(ML_Routines_2D.normalizations, args_norm)
    
    qq = []
    for normval in norms:
        qq.append(normval)
    qq = np.array(qq)
    
    #######################################################

    loglik = np.log(likvec / qq) 
        
    print('-Log Likelihood:', -np.sum(loglik))
    print('Fit Parameters:', mbreak, alpha, beta, gamma, hscale0, dnorm0)
    
    return -np.sum(loglik)

##########################################################################################################################

if __name__ == "__main__":

    initial_guess = [-4.02, 0.66, 2., 2.00]
    res = minimize(loglik, x0 = initial_guess,
                   args = (stars['Imag'], stars['pax'], stars['Imag_err'],
                   stars['pax_err'], stars['sinb']),  method = 'BFGS',
                   options = {'gtol': 1e-3})

    print(res, '\n')
    print('Errors:')
    print(np.sqrt(res.hess_inv))
