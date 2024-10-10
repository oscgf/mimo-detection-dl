# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:31:26 2021

@author: Óscar
"""

import numpy as np
import matplotlib.pyplot as plt
from labcomdig import transmisorpamV2, Qfunct, detectaSBF, simbolobit
from functions import de2Nary

#%% INITIALIZATION

Nb = 1e5                    #Number of bits on transmission
Eb = 9                      #Average energy of the bit
M = 4                       #4-PAM modulation
n = 2                       #Antenna array NxN
SNRmin = 10                 #Minimum SNR [dB]
SNRmax = 36                 #Maximun SNR [dB]
SNRstep = 1                 #Step for SNR array[dB]
k = np.log2(M)              #Number of bits on each symbol
Ns = int(np.floor(Nb/k))    #Number of symbols (multiple of k)
Nb = int(Ns*k)              #Number of bits on transmission (multiple of k)
Es = k*Eb                   #Symbol energy
verbose = 1                 #Progress indicator
Nh = 100                    #Number of Random Channels to average

#%% M-PAM TRANSMITTER

#Data generation as one sequence
#We are supposing a multi-user transmission
bn = np.random.randint(2,size=Nb*n)
[Xn,Bn,An,phi,alphabet] = transmisorpamV2(bn,Eb,M,np.ones(1),1)

#Data reshape. Each row represents an antenna
Xn = Xn.reshape([n,Ns])
An = An.reshape([n,Ns]) 
Bn = Bn.reshape([n,Nb]) 

#%% CHANNEL

SNRdb = np.arange(SNRmin,SNRmax,SNRstep) #SNR values [dB]

#Possible alphabet combinations at reception (needed in ML)
x = alphabet[de2Nary(np.arange(M**n),n,len(alphabet))]

BERav = np.empty([3,len(SNRdb),Nh])

berZFh= np.empty([n,len(SNRdb),Nh])
berLMMSEh= np.empty([n,len(SNRdb),Nh])
berMLh= np.empty([n,len(SNRdb),Nh])

normalization = True #This will avoid the Rayleigh fading behaviour
#Normalization is performed rowwise: rows of H have unit norm.

#%% TRANSMISION AND DETECTION
for jj in range(Nh):
    H = np.random.randn(n,n)
    if normalization:
        H=H/np.linalg.norm(H,axis=0)
        H=H.T
    Hinv = np.linalg.inv(H) #Inverse channel matrix
        
    berZF = np.empty([n,0])
    berML = np.empty([n,0])
    berLMMSE = np.empty([n,0])
    
    if verbose: print('\nSimulating Channel # = ', jj)
    if verbose: print('Simulating SNR = ', end = '')
    for ii in range(len(SNRdb)):
        if verbose: print('{}, '.format(SNRdb[ii]), end = '')
        SNR = 10**(SNRdb[ii]/10)          #Signal to Noise Ratio [n.u.]
        varzn = Eb/(2*SNR)                              #Noise variance
        Wn = np.sqrt(varzn)*np.random.randn(*Xn.shape)  #AWGN
    
        Rn = H@Xn + Wn  #Channel output
        
        # ZF
        Hinv = np.linalg.inv(H)
        Z = Hinv@Rn
        
        Andetected = detectaSBF(Z.flatten(),alphabet).reshape([n,Ns]) #Detected symbols
        Bndetected = simbolobit(Andetected.flatten(),alphabet).reshape([n,Nb]) #Detected bits
        errorsBit = Nb-np.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
        berZF = np.hstack([berZF,(errorsBit/Nb).reshape([n,1])])
        
        # LMMSE
        Z = np.linalg.inv(H.T@H + varzn/Es*np.eye(n))@H.T@Rn
    
        Andetected = detectaSBF(Z.flatten(),alphabet).reshape([n,Ns]) #Detected symbols
        Bndetected = simbolobit(Andetected.flatten(),alphabet).reshape([n,Nb]) #Detected bits
        errorsBit = Nb-np.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
        berLMMSE = np.hstack([berLMMSE,(errorsBit/Nb).reshape([n,1])])
        
        # ML
        hx = np.expand_dims(H@x,axis=2) #New axis
        Z = x[:,np.argmin(np.linalg.norm(Rn.reshape(n,1,Ns)-hx,axis=0),axis=0)]
        
        Andetected = detectaSBF(Z.flatten(),alphabet).reshape([n,Ns]) #Detected symbols
        Bndetected = simbolobit(Andetected.flatten(),alphabet).reshape([n,Nb]) #Detected bits
        errorsBit = Nb-np.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
        berML = np.hstack([berML,(errorsBit/Nb).reshape([n,1])])  
    
    print('\nCalculating simulated bit error rate')
    berZFh[:,:,jj] = berZF
    berLMMSEh[:,:,jj] = berLMMSE
    berMLh[:,:,jj] = berML
    
berZFhav    = np.sum(berZFh,axis=2)/Nh
berLMMSEhav = np.sum(berLMMSEh,axis=2)/Nh
berMLhav    = np.sum(berMLh,axis=2)/Nh  
        
#%% BER

SNRdb2 = np.arange(SNRmax, step=0.01)   #SNR array for theoretical calculus
SNR2 = 10**(SNRdb2/10)                  #SNR in narutal units

print('\n\nCalculating theoretical bit error rate')
serTheo = (2*(M-1)/M)*Qfunct(np.sqrt((6*np.log2(M)/(M**2-1))*SNR2))
berTheo = serTheo/np.log2(M)

#%% RESULTS

print('\nShowing results')
legend = ('ZF','LMMSE','ML','Theoretical SISO')
font = {'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)

# BER per antenna
plt.figure(1,figsize=(10,7))
legend = ('ZF-1','ZF-2','LMMSE-1','LMMSE-2','ML-1','ML-2','Theor. SISO')
plt.semilogy(SNRdb,berZFhav.T,'*',SNRdb,berLMMSEhav.T,'v',SNRdb,berMLhav.T,'x',SNRdb2,berTheo,'-k')
plt.axis([SNRmin,SNRmax,10**-7,1])
plt.grid()
plt.xlabel('Signal to Noise Ratio $E_b/N_0$ (dB)');
plt.ylabel('Bit Error Rate ($P_b$)');
plt.legend(legend)
plt.title('BER per antenna')

plt.show()

# Averaged BER
plt.figure(2,figsize=(10,7))
legend = ('ZF','LMMSE','ML','Theoretical SISO')
plt.semilogy(SNRdb,sum(berZFhav)/n,'*r',SNRdb,sum(berLMMSEhav)/n,'+g',SNRdb,sum(berMLhav)/n,'xb',SNRdb2,berTheo,'-k')
plt.axis([SNRmin,SNRmax,10**-7,1])
plt.grid()
plt.xlabel('Signal to Noise Ratio $E_b/N_0$ (dB)');
plt.ylabel('Bit Error Rate ($P_b$)');
plt.legend(legend)
plt.title('Averaged BER')

plt.show()