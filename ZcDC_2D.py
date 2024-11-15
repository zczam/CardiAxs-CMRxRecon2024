# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:39:55 2021

@author: Zc
"""

import scipy.io as sio
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
import numpy as np

class ZcDC_2D(nn.Module):
 
    # Data consistency part.
    # initialization: give it coil maps, fast operator, and miu.
    # here we set miu is learnable, the given miu is a starting point
    # it solves the following equation:
    # x = (E^H Ex + miu*I)^-1 (E^Hy + miu*z)
    
    #initialize the DC module: ZcDC_2D(miu)
    def __init__(self, miu):
        super().__init__()
        self.miu = nn.Parameter(torch.tensor([miu]), requires_grad=True)
        
    # Forward is a CG solving the objective function. 
    # similar to every CG we have been using so far
    
    # solve CG: x = ZcDC_2D(z, zero-filled)
    # notice the zero-filled and z are 2-channel-splited real 
        
    def forward(self, z, zerofilled, coil, SamplingMask):
        
        # firstly map the real-valed image to complex
        # initialized as p = r = EHy + miu*z
        # reminder: input images in dimension [batch, time, coil, ky, kx], coil = 2 for zf and z as real-imag axis.
        p_now = zerofilled[:,0:1,:,:,:] + zerofilled[:,1:2,:,:,:]*1j + torch.abs(self.miu)*(z[:,0:1,:,:,:] + z[:,1:2,:,:,:]*1j) # (EHE+miuI)x = EHy+miuz, here is EHy+miuz 
        r_now = torch.clone( p_now)
        b_approx = torch.zeros_like(p_now)

        for i in range(5):
            
            q = self.ZcEHEx_rapid(p_now, coil,SamplingMask) + torch.abs(self.miu)*p_now; # (EHE+miuI)x = EHy+miuz, here is (EHE+miuI)x
            rrOverpq = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))  # rrOverpq = (r'*r)/(p'*q);
            b_next = b_approx + rrOverpq*p_now
            r_next = r_now - rrOverpq*q;   
            p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now # p = r_next + ( (r_next'*r_next)/(r'*r) )*p;
            b_approx = b_next
    
            p_now = torch.clone(p_next)
            r_now = torch.clone(r_next)

        return torch.cat([torch.real(b_approx), torch.imag(b_approx)], dim=1)
    
    
    def ZcFFT2D(self,image, axis=[-1,-2]):
    
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(image,dim=axis), dim=axis, norm = 'ortho'), dim=axis)
    
    def ZcIFFT2D(self,kspace, axis=[-1,-2]):
        
        return torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(kspace,dim = axis), dim=axis, norm = 'ortho'),dim = axis)
    

    def ZcEHEx_rapid(self,x, coil, SamplingMask):
         EHEx = torch.sum(self.ZcIFFT2D(  SamplingMask*(self.ZcFFT2D(x*coil))  ) * torch.conj(coil),axis=1,keepdim=True)
         return EHEx
