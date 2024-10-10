# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from labcomdig import transmisorpamV2, Qfunct, detectaSBF, simbolobit
from functions import de2Nary

#%% INITIALIZATION

Nb = 1e6                    #Number of bits on transmission
Eb = 9                      #Average energy of the bit
M = 4                       #4-PAM modulation
n = 2                       #Antenna array NxN
SNRmin = 0                  #Minimum SNR [dB]
SNRmax = 24                 #Maximun SNR [dB]
SNRstep = 1                 #Step for SNR array[dB]
k = np.log2(M)              #Number of bits on each symbol
Ns = int(np.floor(Nb/k))    #Number of symbols (multiple of k)
Nb = int(Ns*k)              #Number of bits on transmission (multiple of k)
Es = k*Eb                   #Symbol energy
verbose = 1                 #Progress indicator

#Data generation as one sequence
#We are supposing a multi-user transmission
bn = np.random.randint(2,size=Nb*n)

#%% M-PAM TRANSMITTER

[Xn,Bn,An,phi,alphabet] = transmisorpamV2(bn,Eb,M,np.ones(1),1)

#Data reshape. Each row represents an antenna
Xn = Xn.reshape([n,Ns])
An = An.reshape([n,Ns]) 
Bn = Bn.reshape([n,Nb]) 

#%% CHANNEL

#Various matrix should be used for comparing the differences between algorithms

H = np.eye(n)          #Channel matrix
#H = np.array([[1, 0.45],[0.25, 1]])
#H = np.array([[1, 0.20],[1, 0.25]])

Hinv = np.linalg.inv(H) #Inverse channel matrix

SNRdb = np.arange(SNRmin,SNRmax,SNRstep) #SNR values [dB]

#%% TRANSMISION AND DETECTION

berZF= np.empty([n,0])
berLMMSE= np.empty([n,0])
berML= np.empty([n,0])

#Possible alphabet combinations at reception
x = alphabet[de2Nary(np.arange(M**n),n,len(alphabet))]

if verbose: print('Simulating SNR = ', end = '')
for ii in range(len(SNRdb)):
    if verbose: print('{}, '.format(SNRdb[ii]), end = '')
    
    SNR = 10**(SNRdb[ii]/10)          #Signal to Noise Ratio [n.u.]
    varzn = Eb/(2*SNR)                              #Noise variance
    Wn = np.sqrt(varzn)*np.random.randn(*Xn.shape)  #AWGN
    
    Rn = H@Xn + Wn  #Channel output
    
    # ZERO FORCING (ZF)
    Z = Hinv@Rn
    
    Andetected = detectaSBF(Z.flatten(),alphabet).reshape([n,Ns]) #Detected symbols
    Bndetected = simbolobit(Andetected.flatten(),alphabet).reshape([n,Nb]) #Detected bits
    errorsBit = Nb-np.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
    berZF = np.hstack([berZF,(errorsBit/Nb).reshape([n,1])])

    # LINEAR MINIMUM MEAN SQUARED ERROR (LMMSE)
    Z = np.linalg.inv(H.T@H + varzn/Es*np.eye(n))@H.T@Rn
    
    Andetected = detectaSBF(Z.flatten(),alphabet).reshape([n,Ns]) #Detected symbols
    Bndetected = simbolobit(Andetected.flatten(),alphabet).reshape([n,Nb]) #Detected bits
    errorsBit = Nb-np.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
    berLMMSE = np.hstack([berLMMSE,(errorsBit/Nb).reshape([n,1])])
    
    # MAXIMUM LIKELIHOOD (ML)
    #Option 1: All at once (faster but expensive in memory)
    hx = np.expand_dims(H@x,axis=2) #New axis
    Z = x[:,np.argmin(np.linalg.norm(Rn.reshape(n,1,Ns)-hx,axis=0),axis=0)]
    
    #Option 2: Column by column (slower but not expensive in memory)
    '''
    Z = np.empty([n,0])
    for column in Rn.T:
        y = np.array([column]).T #Used for operating with the column
        ind = np.argmin(np.linalg.norm(y-H@x,axis=0)) #Index of the minimum
        Z = np.hstack([Z,x[:,ind].reshape(n,1)])
    '''
    
    Andetected = detectaSBF(Z.flatten(),alphabet).reshape([n,Ns]) #Detected symbols
    Bndetected = simbolobit(Andetected.flatten(), alphabet).reshape([n,Nb]) #Detected bits
    errorsBit = Nb-np.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
    berML = np.hstack([berML,(errorsBit/Nb).reshape([n,1])])

#%% BER

SNRdb2 = np.arange(SNRmax, step=0.01)   #SNR array for theoretical calculus
SNR2 = 10**(SNRdb2/10)                  #SNR in narutal units

print('\n\nCalculating theoretical symbol error rate')
serTheo = (2*(M-1)/M)*Qfunct(np.sqrt((6*np.log2(M)/(M**2-1))*SNR2))
berTheo = serTheo/np.log2(M)

print('Calculating simulated bit error rate')
berZFav    = sum(berZF)/n
berLMMSEav = sum(berLMMSE)/n
berMLav    = sum(berML)/n

#%% RESULTS

print('\nShowing results')
font = {'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)

# BER per antenna
plt.figure(1,figsize=(10,7))
legend = ('ZF-1','ZF-2','LMMSE-1','LMMSE-2','ML-1','ML-2','Theor. SISO')
plt.semilogy(SNRdb,berZF.T,'*',SNRdb,berLMMSE.T,'v',SNRdb,berML.T,'x',SNRdb2,berTheo,'-k')
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
plt.semilogy(SNRdb,berZFav,'*r',SNRdb,berLMMSEav,'vg',SNRdb,berMLav,'xb',SNRdb2,berTheo,'-k')
plt.axis([SNRmin,SNRmax,10**-7,1])
plt.grid()
plt.xlabel('Signal to Noise Ratio $E_b/N_0$ (dB)');
plt.ylabel('Bit Error Rate ($P_b$)');
plt.legend(legend)
plt.title('Averaged BER')

plt.show()