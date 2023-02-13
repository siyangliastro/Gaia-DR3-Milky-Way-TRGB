import numpy as np
from scipy.interpolate import CubicSpline
from scipy import special
from astropy.io import ascii
from scipy import interpolate
from scipy.optimize import minimize 
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table
from astropy.table import Column, vstack
import scipy.optimize as optimization
from scipy import stats
import time
from multiprocessing import Pool
import multiprocessing as mp

def plb_js(mag, mbreak=0, alpha=0, beta=0, gamma=0, 
   sig=0, normalize=0):
  
# Double power law model:
# ; AA (m) = 10^(alpha(m-mb)+beta)   m < mb
# ; BB (m) = 10^(gamma(m-mb))        m > mb

    if alpha == 0:
        alpha = 0.3
    if beta == 0:
        beta = 0.3
    if gamma == 0:
        gamma = 0.2
    if sig == 0:
        sig = 0.004
    if mbreak == 0:
        mbreak = -4.0

    x =  mag - mbreak
    aa = 10.0**(alpha*x+beta)
    bb = 10.0**(gamma*x)
    
    plb_js = ((aa-bb)*special.erf(x/sig) + (aa+bb)) / 2.0
    
    if normalize != 0:
#            ; normalize must be set to the normalization range
#    ; if any of the values fall outside this range, that's an error
        if len(normalize) != 2:
            print('NORMALIZE must be given as the normalization range')
            print('Ignoring normalization request')
        else:
            pnorm = plb_js_norm(normalize[0], normalize[1], mbreak, 
            alpha, beta, gamma, sig)
            plb_js = plb_js / pnorm
        #   ; Now check that all values are in normalization range
        if np.min(mag) < normalize[0] or np.max(mag) > normalize[1]:
            print(' ERROR: some given values fall outside normalization range')
            print(' Continue at own risk!')
    return plb_js


def plb_js_noise(mag, mbreak, alpha = 0.3, beta = 0.3, 
   gamma = 0.2, sig = 0.004, noise = 0.1, method = 1, nnodes = 201,
   true_range = [-6.5, -1.5], quiet = 0):

    if alpha == 0:
        alpha = 0.30
    if beta == 0:
        beta = 0.30
    if gamma == 0:
        gamma  = 0.20
    if sig == 0:
        sig = 0.0040
    if true_range == 0: 
        true_range = [-6.5, -1.5]
    if len(true_range) != 2:
        print('Invalid range; must have two elements')
        return 0

    if true_range[1] < true_range[0]:
        print('Invalid range: must be a positive-length interval')
        return 0

    if quiet == 0:
        quiet = 0

#     if noise == 0:
#         if quiet == 0:
#             print('Warning: noise defaults to zero: ')
#             print('   reverting to plb_js with normalization')
#             print('Set /quiet to suppress this message')
#             return plb_js(mag, mbreak=mbreak, alpha=alpha, beta=beta,
#                           gamma=gamma, sig=sig) # ; / pnorm   ; See above for normalization issues.

    if method == 0:
        method = 1
    if method != 1 and len(noise) > 1:
        if quiet == 0:
            print('METHOD=2 cannot be used if NOISE is an array')
            print('Reverting to METHOD=1 (changed upon return)')
        method = 1

    if nnodes == 0:
        nnodes = 201
        
    dlmin = -5.0
    dlmax = 5.0
    dlstep = (dlmax-dlmin) / (nnodes-1.0)
    dl = dlmin + dlstep*np.linspace(0, nnodes-1, nnodes)

    # ; Now select either a uniform set of luminosity values or those
    # ; actually observed

    if method == 1:
        testpoints = mag
        npoints = len(mag)
    else: 
    # ; do not split; will consider splitting later
        npoint = 2001
        mlow = np.min(mag) < true_range[0]
        mhigh = np.max(mag) > true_range[1]
        testpoints = mlow + (mhigh - mlow) * range(npoints) / (npoints - 1.0)

    #    ; The desired integral is:
    #    ;    f(Lobs) = int plb(Ltrue) exp(-(Lobs-Ltrue)^2/2/sigma^2) 
    #    ;              d Ltrue / sqrt(2 pi sigma^2)
    #    ; and needs to be normalized.
    #    ;

    probtest = np.ones(npoints)
    noisevals = noise + np.zeros(len(testpoints))
    #    ; results in a noiseval array with the same length as testpoints
    testpoints = np.array(testpoints)
    for k in range(npoints):
        mvals = testpoints[k] + dl * np.float(noisevals[k])
        wh = np.where((mvals >= true_range[0]) & (mvals <= true_range[1]))
        probtest[k] = np.sum(np.exp(-dl[wh]**2 / 2.0) * plb_js(mvals[wh], mbreak,
          alpha, beta, gamma, sig)) * dlstep / np.sqrt (2.0 * np.pi)

    #    ; * noisevals[k] / sqrt(2*!dpi*noisevals[k]^2)
    #    ; NOISEVALS[k] cancels out in the normalization
    #    ; stop   

    if method == 1:
        prob = probtest
    else:
        prob = CubicSpline(testpoints, probtest, mag)

    prob = prob #; / pnorm  ; Again, see comments above regarding normalization

    return prob

# ;
# ; Compute the probability of observed MAG, PAX given the 
# ; luminosity function LF(MABS) and the density distribution 
# ; D(dist) ~ Dist^2 exp(- Dist/hscale)
# ;
# ; In principle the LF can be added over multiple ranges.
# ; For now we use an unrealistically small range TRUE_RANGE=[-5,-3].
# ; This limits significantly the logarithmic steps in distance.


def likelihoods(mag, pax, emag, epax, sinb, mbreak, alpha, beta, gamma, hscale0, dnorm0, sig = 0, true_range = 0, explore = 0, pax_min_acc = 0, kfactor = 
0, debug = 0, npmag = 201, npdist = 201):

    hscale = [hscale0]
    dnorm = [dnorm0]

    #; Set up defaults for the distribution parameters
    if alpha == 0:
        alpha = 0.30
    if beta == 0:
        beta = 0.30
    if gamma == 0:
        gamm = 0.20
    if sig == 0:
        sig = 0.0040
    if mbreak == 0:
        mbreak = -4.0
    if hscale == 0:
        hscale = 0.25
   
    ndens = len(hscale)
    if ndens == 0:
        dnorm = 1.0 / ndens * np.ones(ndens)
    if true_range == 0:
        true_range = [-6.5, -1.5]
    if pax_min_acc == 0:
        pax_min_acc = 1./(10 * np.max(hscale)) #; This is the farthest parallax needs to be considered (realistically)
    
    npmag = npmag
    npdist = npdist
    mabs_min = true_range[0]
    mabs_max = true_range[1]
    magstep = (mabs_max - mabs_min) / (npmag - 1.0)
    mabs = mabs_min + magstep * np.arange(0, npmag)

    lf = plb_js_noise(mabs, mbreak = mbreak, alpha = alpha, beta = beta, gamma = gamma, sig = sig, noise = emag,
                      method = 1, nnodes = 201, true_range = [-6.5, -1.5], quiet = 0)#, normalize = 0)
#   ; Normalization must NOT be included here!

    kfactor = 5.0
    mags_int = np.zeros(npmag)
    dist_int = np.zeros([npmag, npdist])
    dist_val = np.zeros([npmag, npdist])
    totint = 0

    mag_dist_range = np.zeros([2, npmag])
    final_dist_range = np.zeros([2, npmag])
    range_extended = np.zeros(npmag)
    
    for kmag in range(npmag):

        this_mabs = mabs[kmag]

        dmod_min = mag - this_mabs - kfactor * emag 
            #; this is the smallest 
            #; value of DMOD consistent with photometry at a K*SIGMA level
        dmod_max = mag - this_mabs + kfactor * emag 
            #; this is the largest 
            #; value of DMOD consistent with photometry at a K*SIGMA level

        dlog_min = dmod_min * 0.2 - 2   #; Decimal log of minimum distance in kpc
        dlog_max = dmod_max * 0.2 - 2   #; Decimal log of maximum distance in kpc  

        mag_dist_range[:, kmag] = [dlog_min, dlog_max]
            #; This is the distance range compatible with photometry
            
            #; Ensure it is also compatible with astrometry
        pax_min = (pax - kfactor * epax) #> pax_min_acc
        if pax_min <= 0:
            pax_min = pax_min_acc
            #; Realistically,
            #; no parallax beyond 1/pax_min_acc (defaults to 10*hscale)
            #; needs to be considered  
        
        pax_max = (pax + kfactor * epax) #< 50.0
            #; Realistically,
            #; no star within 20 pc needs to be considered
        if pax_max >= 50:
            pax_max = 50
        dpax_min = 1.0 / pax_max
        dpax_max = 1.0 / pax_min

        #; Extend distance range to ensure peak is considered
        if dlog_min < np.log10(dpax_min):
            dlog_min = dlog_min
        else:
            dlog_min = np.log10(dpax_min)
        if dlog_max > np.log10(dpax_max):
            dlog_max = dlog_max
        else:
            dlog_max = np.log10(dpax_max)
       # dlog_min = np.log(dpax_min) # dlog_min should be -0.14995991811096587
       # dlog_max = np.log(dpax_max)
#         idx_min = np.where(np.log10(dpax_min) > dlog_min)
#         dlog_min = np.log10(dpax_min)[idx_min]
#         idx_max = np.where(np.log10(dpax_max) > dlog_max)
#         dlog_max = dlog_min < np.log10(dpax_min)

#     #     idx_max = np.where(np.log10(dpax_max) > dlog_max)
#     #     dlog_max = np.log10(dpax_max)[idx_max]
#         #dlog_max = dlog_max > np.log10(dpax_max)

#         final_dist_range[:, kmag] = [dlog_min, dlog_max]

#         if (final_dist_range[0,kmag] != mag_dist_range[0,kmag] or 
#         final_dist_range[1,kmag] != mag_dist_range[1,kmag]):
#               range_extended[kmag] = 0xff
                
        dlogstep = (dlog_max - dlog_min) / (npdist - 1.0) 
        #; (Decimal) logarithmic step in distance

        dlogtrue = dlog_min + dlogstep * np.arange(0, npdist)
        dtrue = 10 ** dlogtrue
        dist_val[kmag, :] = dtrue[:]
        
        paxtrue = 1.0 / dtrue  #; true parallax in mas
        dmod = 5 * dlogtrue + 10   # Deubgging good to here
        
        integral = 0.
        for idens in range(ndens):
            exponent = -(paxtrue-pax)**2/2/epax**2 - (mag-(this_mabs+dmod))**2/2/emag**2 \
            - dtrue * sinb/hscale[idens] + 3*np.log(dtrue)         

    #    ; This is the natural log of the integrand, containing most 
    #    ; of the rapidly varying quantities.  Only consider the integration
    #    ; steps where the integrand is sufficiently close to its maximum.     

            wh = np.where(exponent > (max(exponent) - 20))  
            dist_int[kmag, wh] += np.exp(exponent[wh]) * dlogstep / (2*np.pi*emag*epax)
            integral += dnorm[idens] * np.sum(np.exp(exponent[wh]) * dlogstep / (2*np.pi*emag*epax))
            #print(integral)
            #np.sum(np.exp(exponent[wh]) * dlogstep) / (2*emag*epax)
    #    ; normalization may not be necessary
    #    ; and wastes a very small amount of
    #    ; time, but it helps clarity
        mags_int[kmag] = integral
        totint = totint + integral * magstep * lf[kmag]
        #; If explore is set, stop to allow looking at the various distributions
    #     if (explore != 0):
    #         break
        if debug != 0 and np.sum(range_extended > 0):
            print('Range extended ', mag, pax, np.sum(range_extended > 0))
    return totint


def normalizations(emag, epax, sinb, mbreak, alpha, beta, gamma, hscale0, dnorm0, 
                   absrange = [-5.0, -3.0], true_range = [-6.5, -1.5], magrange = [8.0, 11.7], 
                   magstep = .02, paxstep = .02, paxrange = [1.0, 5.0], details = 0):
    
# ;
# ; This function computes the normalization integral over a
# ; magnitude and parallax domain defined as follows:
# ; mag in [magmin, magmax]
# ; pax in [paxmin(mag), paxmax(mag)]
# ; i.e., the minimum an dmaximum parallax are a function of
# ; magnitude
# ;
# ; The primary application to the case of the TRGB in the Milky Way
# ; has the minimum and maximum parallaxes at each magnitude
# ; defined as the parallaxes at which the absolute magnitude is in a
# ; specific range.  This can be provided as parameter ABSRANGE.
# ; However, the routine will also accept an external function defining
# ; PAXRANGE as a function of magnitude.
# ;

    if paxrange == 0:
        use_function = 1
    else: 
        use_function = 0
        if absrange == 0:
            absrange = [-5.0, -3.0]

    # ; Define the (apparent) magnitude values over which to carry the
    # ; integral over parallax    

    nmag = round((np.max(magrange) - np.min(magrange)) / magstep + 1)# > 3
    if nmag <= 3:
        nmag = 5
    if ((nmag % 2) == 0):
        nmag = nmag + 1  # ; ensure nmag is odd

    actual_magstep = (np.max(magrange) - np.min(magrange)) / (nmag - 1)
    magvals = np.min(magrange) + actual_magstep * (np.arange(nmag))
    nmag = int(nmag)

    # Simpson Rule Weights
    wmag = (4.0 / 3.0) * np.ones(nmag)
    wmag[2:nmag-1:2] = 2.0 / 3.0
    wmag[0] = 1.0 / 3.0
    wmag[nmag-1] = 1.0 / 3.0
    wmag = wmag * actual_magstep

    # ; Odd number of points covering the mag interval with equal steps.

    # ; Now step over apparent magnitude intervals and carry out the
    # ; integral oiver parallax.  Save each integral separately (possibly
    # ; for later smoothness tests).  Define the minimum and maximum
    # ; parallax value at each value of apparent magnitude.

    minpax = np.ones(nmag)
    maxpax = np.ones(nmag)

    # ; Note: we want an odd number of parallax steps to cary out Simpson
    # ; integration.  

    if (use_function == 1):
        for k in range(nmag):
            range1 = paxrange(magvals[k])
            minpax[k] = range1[0]
            maxpax[k] = range1[1]
    else:
        dmodmax = magvals - np.min(absrange)
        dmodmin = magvals - np.max(absrange)
        distmax = 10.0 ** (0.20 * dmodmax - 2.0)
        distmin = 10.0 ** (0.20 * dmodmin - 2.0)
        maxpax = 1.0 / distmin
        minpax = 1.0 / distmax

    # ; Now we have a minimum and maximum parallax for each apparent magnitude.
    # ; Define the parallax values over which the integration is carried
    # ; out.

    intpax = np.ones(nmag)  #   ; holds the value of the integral over
                            #   ; parallax for each magnitude value.
    npax = np.ones(nmag)
    actual_paxstep = np.ones(nmag)

    for k in range(nmag):
    #  ; find the actual number of steps and parallax step to be used.
    #  ; The number of steps must be an odd integer
        npax[k] = round(((maxpax[k] - minpax[k]) / paxstep + 1))# > 3

        if ((npax[k] % 2) == 0):
            npax[k] += 1
        if npax[k] <=3:
            npax[k] = 3
        actual_paxstep[k] = ((maxpax[k] - minpax[k]) / (npax[k] - 1))

    # ; The lines defining the lists of arrays are only for debugging purposes
    paxvals_list = []
    probvals_list = []
    wpax_list = []

    # ; Now do the actual integration over parallax
    for k in range(nmag):
        paxvals = minpax[k] + actual_paxstep[k] * np.arange(int(npax[k]))
        paxvals_list.append(paxvals)

        # ; Make the Simpson rule weights
        wpax = (4.0 / 3.0) * np.ones(int(npax[k]))
        wpax[2:int(npax[k])-1:2] = 2.0 / 3.0
        wpax[0] = 1.0 / 3.0
        wpax[int(npax[k]) - 1] = 1.0 / 3.0   
        wpax = wpax * actual_paxstep[k]
        wpax_list.append(wpax)

        #; Evaluate the function
        probvals = np.ones(int(npax[k]))
        for l in range(int(npax[k])):
 
            probvals[l] = likelihoods(magvals[k], paxvals[l], emag, epax, sinb, mbreak, alpha, beta, gamma, hscale0, 
                                                             dnorm0, true_range = true_range)

        probvals_list.append(probvals)    
        intpax[k] = np.sum(wpax * probvals)

    integral = np.sum(wmag * intpax)

    # details = {magvals:magvals, wmag:wmag, actual_magstep:actual_magstep, $
    #            minpax:minpax, maxpax:maxpax, npax:npax, $
    #            actual_paxstep:actual_paxstep, paxvals_list:paxvals_list, $
    #            probvals_list:probvals_list, wpax_list:wpax_list, $
    #            intpax:intpax, integral:integral}

    return integral
