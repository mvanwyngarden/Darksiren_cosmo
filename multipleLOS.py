import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy.special import erf
from scipy.interpolate import interp1d
from tqdm import tqdm
import scipy
import pdb

import seaborn as sns
pal=sns.color_palette('colorblind')

import matplotlib as _mpl

_mpl.rcParams['figure.figsize']= (3.3, 2.5)
_mpl.rcParams['figure.dpi']= 300
_mpl.rcParams['axes.labelsize']= 7
_mpl.rcParams['xtick.labelsize']= 7
_mpl.rcParams['ytick.labelsize']= 7
_mpl.rcParams['legend.fontsize']= 7
_mpl.rcParams['font.size']= 7
_mpl.rcParams['font.family']= 'sans-serif'
_mpl.rcParams['font.sans-serif']= ['DejaVu Sans', 'Arial', 'Helvetica', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Avant Garde', 'sans-serif']
_mpl.rcParams['mathtext.fontset']='dejavusans'
_mpl.rcParams['axes.linewidth']= 0.5
_mpl.rcParams['grid.linewidth']= 0.5
_mpl.rcParams['lines.linewidth']= 1.
_mpl.rcParams['lines.markersize']= 3.
_mpl.rcParams['savefig.bbox']= 'tight'
_mpl.rcParams['savefig.pad_inches']= 0.01
#mpl.rcParams['text.latex.preamble']= '\usepackage{amsmath, amssymb, sfmath}'

def normal_distribution(x,mu,sigma):
    '''
    Reuturns a simple gaussian likelihood:
    
    Parameters
    ----------
    x: np.array
        Where to evaluate likelihood
    mu,sigma: np.array
        Mean and standard deviation of gaussian
    '''
    var=sigma**2
    return np.power(2*np.pi*var,-0.5)*np.exp(-0.5*np.power((x-mu)/sigma,2.))

def normal_distribution_inco(x,mu,sigma):
    '''
    Reuturns a gaussian likelihood without overall normalization:
    
    Parameters
    ----------
    x: np.array
        Where to evaluate likelihood
    mu,sigma: np.array
        Mean and standard deviation of gaussian
    '''
    var=sigma**2
    return np.exp(-0.5*np.power((x-mu)/sigma,2.))

def GW_detection_probability(dltrue,sigmadl,dlthr):
    '''
    Return a detection probability given a gaussian likelihood. The detection probability is defined as 
    the integral from -inf to x_thr of a given normal distribution with mean mu and std.
    
    Parameters 
    ----------
    dltrue: np.array
        Where to evaluate the det prob
    sigmadl: np.array
        std of the likelihood model
    dlthr: np.array
        Cut for the selection
    '''
    t_thr= (dlthr-dltrue)/sigmadl # Normalized random variable
    return 0.5*(1+erf(t_thr/np.sqrt(2))) # see equation 22

def build_interpolant(z_obs,sigmazevalobs,zrate,nocom='dvcdz'):
    '''
    This function returns the p(z|c) interpolator assuming constant rate
    
    Parameters
    ----------
    z_obs: array
        List of observed values for galaxies
    sigmazevalobs: array
        List of uncertainties on observed redshifts
    z_rate: float
        Maximum redshift for the rate
    '''
    #make an array of possible z values
    zinterpolant=np.exp(np.linspace(np.log(1e-4),np.log(zrate),80000))
    
    #list to hold interpolated z values
    interpolant=np.zeros_like(zinterpolant)
    cosmo=FlatLambdaCDM(H0=70.,Om0=0.25)
    #uncertainty on redshift
    sigmaz=0.013*np.power(1+zinterpolant,3.)
    #set max uncertainty
    sigmaz[sigmaz>0.015]=0.015
       
    # This is the prior to applied to the interpolant 
    if nocom=='uniform':
        dvcdz_ff=1.
    elif nocom=='dvcdz':
        dvcdz_ff=cosmo.differential_comoving_volume(zinterpolant).value #assuming galaxy distribution is uniform over a comoving volume
    else:
        gkde=scipy.stats.gaussian_kde(nocom,bw_method=0.8e-1)
        dvcdz_ff=gkde(interpolant)
        
    # loop through all observed redshifts, constructing a posterior for each observed redshift
    for i in tqdm(range(len(z_obs))):
        # Initializes array for the calculation of the interpolant (the z values we are interpolating between)
        zmin=np.max([0.,z_obs[i]-5*sigmazevalobs[i]]) #set up to ensure we don't get a negative lower bound on z
        zeval=np.linspace(zmin,z_obs[i]+5*sigmazevalobs[i],5000) #generate 5000 z values between lower and upper bound
        sigmazeval=0.013*np.power(1+zeval,3.) #calculate uncertainty for each value
        sigmazeval[sigmazeval>0.015]=0.015
        
        if nocom=='uniform':
            dvcdz=1.
        elif nocom=='dvcdz':
            dvcdz=cosmo.differential_comoving_volume(zeval).value #make the prior corresponding to the 5000 z's
        else:
            dvcdz=gkde(zeval)
        
        # The line below is the redshift likelihood times the prior. Assuming a gaussian distribution for the evaluated redshifts.
        pval=normal_distribution(zeval,mu=z_obs[i],sigma=sigmazeval)*dvcdz #creates the posterior for the z of interest 
        normfact=scipy.integrate.simpson(pval,x=zeval) # Renormalize   (denominator of Eq 16)
        
        evals=normal_distribution(zinterpolant,mu=z_obs[i],sigma=sigmaz)*dvcdz_ff/normfact
        if np.all(np.isnan(evals)):
            continue
        interpolant+=evals # Sum to create the interpolant 
        
    interpolant/=scipy.integrate.simpson(interpolant,x=zinterpolant) 
    return interp1d(zinterpolant,np.log(interpolant),bounds_error=False,fill_value=-np.inf),zinterpolant

def draw_gw_events(sigma_dl,dl_thr,galaxies_list,true_cosmology,zcut_rate):
    '''
    This function draws the GW events and applies a selection based on the observed luminosity distance
    
    Parameters
    ----------
    Ndet: Number of GW detections you want
    sigma_dl: Fractional value of the std for the GW likelihood
    dl_thr: Threshold for detection in Mpc
    galaxies_list: array with the redshift of the galaxies
    true_cosmology: Astropy FlatLambdaCDM class for the true cosmology
    zcut_rate: until what redshift GW events to happen
    
    Returns
    -------
    Observed luminosity distance in Mpc, True luminosity distance in Mpc, True GW redshift, Standard deviation used to draw sample
    '''
    Ngw=10000
    rate_term = np.zeros_like(galaxies_list) # Initialize the rate term
    #above makes an array of zeros, next line turns the zero to one for any redshifts below the redshift cutoff
    rate_term[galaxies_list<=zcut_rate]=1. 

     # Draw random redshifts from the galaxy list
     #the p gives the probability associated with each entry in galaxies_list, this applies the rate cut (p=0 for redshifts above the cutoff)
    gw_redshift=np.random.choice(galaxies_list,size=Ngw,p=rate_term/rate_term.sum()) 
    # gw_redshift=np.random.choice(galaxies_list,size=Ngw,p=luminosities/luminosities.sum()) 
    #calculate the true luminosity distances for each gw redshift
    gw_true_dl=true_cosmology.luminosity_distance(gw_redshift).to('Mpc').value
    std_dl=gw_true_dl*sigma_dl # Defines the sigma for the gaussian
    gw_obs_dl=np.random.randn(len(gw_true_dl))*std_dl+gw_true_dl # Generate an observed value, a.k.a. data
    gw_detected =  np.where((gw_obs_dl<dl_thr) & (gw_obs_dl>0) )[0] # Finds the detected GW events (detected if dl<threshold)
    gw_detected = gw_detected[:1:]
    return gw_obs_dl[gw_detected], gw_true_dl[gw_detected], gw_redshift[gw_detected],std_dl[gw_detected]

def galaxy_catalog_analysis_photo_redshift(H0_array,zinterpo,gw_obs_dl,sigma_dl,dl_thr):
    '''
    This function will perform the H0 analysis assuming errors on the redshift determination.
    
    Parameters
    ---------
    H0_array: grid of H0 for the analysis
    zinterpo: Interpolant for p(z|C)
    gw_obs_dl: Array containing observed values for the GW luminosity distance in Mpc
    sigma_dl: Array of sigma for dl (in Mpc) used to draw gw_obs_dl
    dl_thr: Threshold for detection in Mpc
    
    Returns
    -------
    Posterior matrix (raws events, columns H0) and combined posterior
    '''

    cosmotrial=FlatLambdaCDM(H0=70.,Om0=0.25) #Om0=omega matter
    #make a 2D matrix with rows for every obs dl and columns for each H0 guess in our H0 prior
    posterior_matrix = np.ones([len(gw_obs_dl),len(H0_array)])
    # Evaluated d_L times H0
    #find dL using the z posterior value and multiply it by real H0
    dltimesH0=cosmotrial.luminosity_distance(zinterpo.x).to('Mpc').value*cosmotrial.H0.value

    selection_bias=np.zeros_like(H0_array)
    
    #zinterpo.x=zinterpolant, so the followling line returns the interpolated redshift posterior (i.e. pcbc)
    pzeval=np.exp(zinterpo(zinterpo.x)) #take the exponent bc the interpolated function is a log 
    
    
    # Calculate the selection effect on the grid just once
    for j,H0 in tqdm(enumerate(H0_array)):
        dltrial=dltimesH0/H0 #luminosity distance based on H0 guess
        #Calculate GW detection prob based on trial dl
        integrand=GW_detection_probability(dltrial,sigmadl=sigma_dl*dltrial,dlthr=dl_thr)*pzeval #denominator of Eq 3
        selection_bias[j]=scipy.integrate.simpson(integrand,x=zinterpo.x)
    
    # Loop on the GW events and H0 to compute posteriors
    for i,idx in tqdm(enumerate(gw_obs_dl),desc='Running on GW events'):
        for j,H0 in enumerate(H0_array):
            dltrial=dltimesH0/H0
            integrand=normal_distribution(gw_obs_dl[i],mu=dltrial,sigma=sigma_dl*dltrial)*pzeval
            # Do the integral of the numerator of Eq 3 in redshift
            numerator = scipy.integrate.simpson(integrand,x=zinterpo.x)
            posterior_matrix[i,j]=numerator/selection_bias[j] # Eq 3
    
    # Combine the result
    # combined=np.ones_like(H0_array)
    # for i,idx in enumerate(gw_obs_dl):
    #     posterior_matrix[i,:]/=scipy.integrate.simpson(posterior_matrix[i,:],x=H0_array)
    #     combined*=posterior_matrix[i,:]
    #     combined/=scipy.integrate.simpson(combined,x=H0_array)

    return posterior_matrix[0,:]

def galaxy_catalog_analysis_accurate_redshift(H0_array,galaxies_list,zcut_rate,gw_obs_dl,sigma_dl,dl_thr):
    '''
    This function will perform the H0 analysis in the limit that the redshift estimate from the catalog
    is without uncertainties
    
    Parameters
    ---------
    H0_array: grid of H0 for the analysis
    galaxies_list: array with the redshift of the galaxies
    zcut_rate: until what redshift GW events to happen
    gw_obs_dl: Array containing observed values for the GW luminosity distance in Mpc
    sigma_dl: Fractional value of the std for the GW likelihood
    dl_thr: Threshold for detection in Mpc
    
    Returns
    -------
    Posterior matrix (raws events, columns H0) and combined posterior
    '''
    
    #initialize array with rows corresponding to GW event and columns for each H0 value in the prior
    posterior_matrix = np.ones([1,len(H0_array)])
    rate_term = np.zeros_like(galaxies_list)
    #set galaxies below the z cutoff equal to one
    rate_term[galaxies_list<=zcut_rate]=1.
    p_z_given_C=rate_term/rate_term.sum() # Calculates what galaxies have a non-zero rate (how many gals we really have to work with)
    cosmotrial=FlatLambdaCDM(H0=70.,Om0=0.25)
    dltimesH0=cosmotrial.luminosity_distance(galaxies_list).to('Mpc').value*cosmotrial.H0.value # Initialize dltimes H0

    for j,H0 in tqdm(enumerate(H0_array),desc='running on H0'):
        dltrial=dltimesH0/H0 # Calculations of dl
        # Selection bias
        selection_bias=np.sum(GW_detection_probability(dltrial,sigmadl=sigma_dl*dltrial,dlthr=dl_thr)*p_z_given_C)
        for i,idx in enumerate(gw_obs_dl):
            # Integral at the numerator (it is a sum since we measure perfectly redshift)
            numerator = np.sum(normal_distribution(gw_obs_dl[i],mu=dltrial,sigma=sigma_dl*dltrial)*p_z_given_C)
            posterior_matrix[i,j]=numerator/selection_bias

    return posterior_matrix[0,:]

def galaxy_catalog_analysis_accurate_redshift_oneline(H0_array,galaxies_list,zcut_rate,gw_obs_dl,sigma_dl,dl_thr):
    '''
    This function will perform the H0 analysis in the limit that the redshift estimate from the catalog
    is without uncertainties
    
    Parameters
    ---------
    H0_array: grid of H0 for the analysis
    galaxies_list: array with the redshift of the galaxies
    zcut_rate: until what redshift GW events to happen
    gw_obs_dl: Array containing observed values for the GW luminosity distance in Mpc
    sigma_dl: Fractional value of the std for the GW likelihood
    dl_thr: Threshold for detection in Mpc
    
    Returns
    -------
    Posterior matrix (raws events, columns H0) and combined posterior
    '''
    
    #initialize array with rows corresponding to GW event and columns for each H0 value in the prior
    posterior_matrix = np.ones([len(gw_obs_dl),len(H0_array)])
    rate_term = np.zeros_like(galaxies_list)
    #set galaxies below the z cutoff equal to one
    rate_term[galaxies_list<=zcut_rate]=1.
    p_z_given_C=rate_term/rate_term.sum() # Calculates what galaxies have a non-zero rate (how many gals we really have to work with)
    cosmotrial=FlatLambdaCDM(H0=70.,Om0=0.25)
    dltimesH0=cosmotrial.luminosity_distance(galaxies_list).to('Mpc').value*cosmotrial.H0.value # Initialize dltimes H0

    for j,H0 in tqdm(enumerate(H0_array),desc='running on H0'):
        dltrial=dltimesH0/H0 # Calculations of dl
        # Selection bias
        selection_bias=np.sum(GW_detection_probability(dltrial,sigmadl=sigma_dl*dltrial,dlthr=dl_thr)*p_z_given_C)
        for i,idx in enumerate(gw_obs_dl):
            # Integral at the numerator (it is a sum since we measure perfectly redshift)
            numerator = np.sum(normal_distribution(gw_obs_dl[i],mu=dltrial,sigma=sigma_dl*dltrial)*p_z_given_C)
            posterior_matrix[i,j]=numerator/selection_bias

    # Normalize and combine posteriors
    combined=np.ones_like(H0_array)
    for i,idx in enumerate(gw_obs_dl):
        posterior_matrix[i,:]/=scipy.integrate.simpson(posterior_matrix[i,:],x=H0_array)
        combined*=posterior_matrix[i,:]
        combined/=scipy.integrate.simpson(combined,x=H0_array)
        
    return posterior_matrix,combined

