import numpy as np
import torch
def b_motion_np(n,r,m,sig,S0,K,T,seeded=False):
  """ 
    simulation stock prices according to a geometric. Steps are as follows:
      1) partition the total time interval into equally spaced intervals of length dt
      2) generate wiener processes over the m x n matrix (where n=numb of paths, m=numb of subdivisions in each path)
      3) use anithetic variables to conserve the amount of random shocks simulated
  """
  dt= T/m #partition the total time interval into equally spaced intervals of length dt
  vol = sig*np.sqrt(dt)
  if seeded == True:
    seed= 1234
    rg = np.random.RandomState(seed) #check scale
    z_pos = rg.standard_normal(size = (int(m), int(n/2)))
  else:
    z_pos = np.random.standard_normal(size = (int(m), int(n/2)))
  z_neg = np.negative(z_pos)                                                  
  incs = np.concatenate((z_pos, z_neg),axis=1)
  incs_cumsum =  np.concatenate((np.zeros((1,n)),incs),axis=0).cumsum(axis=0) 
  incs_cumsum *= vol
  tline = np.linspace(0,T,m+1)                            
  t_mat =  np.repeat(tline.reshape((m+1,1)), n, axis=1)   
  drift_cumsum = (r - 0.5*sig**2) * t_mat
  St = S0 * np.exp(incs_cumsum + drift_cumsum)            
  return St, t_mat
  
  
    #Torch Price Simulation
def b_motion_torch(self, n, r, m, sig, S0, K, T, seeded=False ):
    dt= T/m 
    vol = sig*np.sqrt(dt)
    if seeded == True:
      torch.manual_seed(1234)
      z_pos = torch.normal(0,1, size=(int(m), int(n/2)) )
    else:
      z_pos = torch.normal(0,1, size=(int(m), int(n/2)) )
    z_neg = torch.negative(z_pos)                                                                           
    incs = torch.cat((z_pos, z_neg),1 )
    incs_cumsum =  torch.cat((torch.zeros((1,n)),incs),0)
    incs_cumsum = torch.cumsum(incs_cumsum,0)
    incs_cumsum *= vol
    tline = torch.linspace(0,T,m+1)                            
    t_mat =  torch.repeat_interleave(tline.reshape((m+1,1)), n, dim=1)   
    drift_cumsum = (r - 0.5*sig**2) * t_mat        
    St = S0 * torch.exp(incs_cumsum + drift_cumsum)
    return St, t_mat