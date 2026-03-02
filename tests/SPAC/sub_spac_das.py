import numpy as np
import matplotlib.pyplot as plt
from scipy.special import *

class fit_spac_strain():

    def exx_exx_spac(self,cr,cl,f,dist,HV,AR,AL):

        kr =  (2.0 * np.pi * f )/cr
        kl = (2.0 * np.pi * f )/cl

        xr = dist * kr
        xl = dist * kl
        
        coef_r = ( AR * (HV**2) ) /8
        spac_r =  (kr**2)  * ( 3.0 * jn(0, xr)  - 4.0* jn(2, xr) + jn(4, xr) ) 

        # LOVE
        coef_l = (AL ) /8
        spac_l = (kl**2) * ( jn(0, xl) - jn(4, xl) )

        spac_R =  coef_r * spac_r
        spac_L = coef_l * spac_l
        spac = spac_R + spac_L

        return spac,spac_R,spac_L,coef_r,coef_l

    def exx_ray_spac_nocoef(self,f,dist,cr):
        kr =  (2.0 * np.pi * f )/cr
        xr = dist * kr

        ray_spac_nocoef =  (kr**2)  * ( 3.0 * jn(0, xr)  - 4.0* jn(2, xr) + jn(4, xr) ) 

        return ray_spac_nocoef, kr
    #----------------------------------
    # <INPUT>
    # f0 : frequencey
    # cross_obs: Real part ross-spectrum at frequancy of f0
    # phaselist: numpy array of Phasevelocity (search range)
    # distlist: numpy array of interstaion distance
    #----------------------------------
    def grid_search_fitting_rayleigh(self,f,cross_obs,distlist,phase_search):

        nccr = len(cross_obs)
        G = np.zeros([nccr,1],dtype=np.float64)

        ngrid = len(phase_search)
        cost_list = np.zeros([ngrid],dtype=np.float64)
        Amp_search = np.zeros([ngrid],dtype=np.float64)
        syn_spac_search = np.zeros([ngrid,nccr],dtype=np.float64)
        for ip in range(ngrid):


            cR = phase_search[ip]
            #------------------------------------       
            # MK G
            #------------------------------------ 

            ray_spac_nocoef,kr = self.exx_ray_spac_nocoef(f,distlist,cR)
            G[:,0] = ray_spac_nocoef

            #------------------------------------       
            # Inverted A and calc syn ray_spac
            #------------------------------------ 

            A = np.linalg.lstsq(G, cross_obs,rcond=None)[0]
            Amp_search[ip] = A[0]
            syn_ray_spac = np.dot(G,A)
            
            syn_spac_search[ip,:] = syn_ray_spac[:]
            
            delta =( cross_obs  - syn_ray_spac) 

            cost = np.dot(delta,delta)/nccr 
            cost_list[ip] = cost

        #-----------------------------------------
        # return data
        #-----------------------------------------
        
        return cost_list,Amp_search,syn_spac_search

    #-----------------------------------------
    # only fit Ampilithde
    #-----------------------------------------       
    def fitting_Amp(self,f,spac,cR0,distlist):
    
        nccr = len(spac)
        G = np.zeros([nccr,1],dtype=np.float64)

        cR = cR0 

        #------------------------------------       
        # MK G
        #------------------------------------ 

        ray_spac_nocoef = self.exx_ray_spac_nocoef(f,distlist,cR)
        G[:,0] = ray_spac_nocoef

        #------------------------------------       
        # Inverted A and calc syn ray_spac
        #------------------------------------ 

        A = np.linalg.lstsq(G, spac,rcond=None)[0]
        syn_ray_spac = np.dot(G,A)        
        delta = spac  - syn_ray_spac

        cost = np.dot(delta,delta)/nccr
        #-----------------------------------------
        # return data
        #-----------------------------------------
        print(A)
        exit()
        
        return A
        