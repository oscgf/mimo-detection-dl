#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Conjunto de funciones desarrolladas para ser utilizadas en los programas principales
'''
import numpy as np
import torch
import matplotlib.pyplot as plt

from labcomdig import gray2de
from network import *

'''
NUMPY
'''
def transmisorpamV2(Bn,Eb,M,p,L):
    """
    [Xn,Bn,An,phi,alfabetopam] = transmisorpamV2(Bn,Eb,M,p,L) 
    
    Entradas:    
     Bn = Secuencia de dí­gitos binarios
     Eb = Energía media por bit transmitida en Julios
     M  = Número de síímbolos del código PAM
     p  = Pulso paso de baja o paso de banda 
     
     L  = Número de puntos a utilizar en la representación de un sí­mbolo
    
    Devuelve:
     Xn = la señal de información (discreta)
     Bn = La secuencia de dí­gitos binarios realmente transmitidos
     An = La secuencia de niveles de amplitud transmitidos
   	 phi = Pulso básico real normalizado (energí­a unidad)
     alfabetopam = Los niveles de amplitud asociados a cada sí­mbolo
    """  
    # Comprobación de parámetros de entrada
    p=np.reshape(p,[np.size(p)])  # <=   #p=p.squeeze()
    if len(Bn)<1 or Eb<=0 or M<2 or np.dot(p,p)==0 or L<1:
        raise Exception('Error: revise los parámetros de entrada')
    # Se obtienen en primer lugar los niveles asociado a cada sí­mbolo ¿Cuántos bits hay en cada sí­mbolo?
    k = int(np.ceil(np.log2(M)))   
    M = 2**(k) # Se Ajusta M a una potencia de dos
    
    # El alfabeto [Ver la ecuación (4.21)] 
    alfabetopam = np.sqrt(3*Eb*np.log2(M)/(M**2-1))*(2*(np.arange(M))-M+1)
    
    # Si Bn no tiene una longitud múltiplo de k, se completa con ceros
    Nb = len(Bn)  # Número de bits a transmitir, actualizado
    Bn = Bn.squeeze().astype(int) #Comprobación de int y dimensiones   
    Bn = np.r_[Bn,np.zeros(int(k*np.ceil(Nb/k)-Nb)).astype(int)] #
    Nb = len(Bn)  # Número de bits a transmitir tras la corrección
    Ns = Nb//k # Número de sí­mbolos a transmitir
    
    # La secuencia generada
    if M>2:
        An = alfabetopam[gray2de(np.reshape(Bn,[Ns,k]))]
    else:
        An = alfabetopam[Bn]
    
    # Comprobación de las longitudes y otros datos del pulso suministrado para 
    # hacer que el número de muestras del mismo sea efectivamente L
    Ls = len(p)
    if Ls<L:
        p = np.r_[p, np.zeros(L-Ls)]
    elif Ls>L:
        print('La duración del pulso se trunca a {} muestras'.format(str(L)))
        p = p[:L] #Debe modificarse si se quiere un pulso de más de L muestras 
    # Se normaliza la energí­a del pulso para obtener la base del sistema
    phi = p / np.sqrt(p@p) 
       
    # Obtención del tren de pulsos, Xn = np.kron(An,phi) ó
    Xn = np.reshape(np.reshape(An,[Ns,1])*phi,[Ns*L,]) #Debe modificarse si se quiere un pulso de más de L muestras
    return [Xn,Bn,An,phi,alfabetopam]

def de2Nary(d,n,N):
    """
    b = de2Nary(d,n,N)
    Convierte un número decimal, d, en un vector binario, b, de longitud n
    con base N
    """   
    c = np.zeros([len(d),int(n)])
    for i in range(int(n)): d, c[:,i] = np.divmod(d,N)
    c = np.fliplr(c);
    return c.astype(int).T


'''
TORCH
'''


def t_Qfunct(x):
    """ 
     y = Qfunct(x) evalúa la función Q en x.
    Donde y = 1/sqrt(2*pi) * integral desde x hasta inf de exp(-t^2/2) dt
    """
    y=(1/2)*torch.special.erfc(x/(torch.tensor(2)**.5)) 
    return y
      
def t_gray2de(b):
    """
    Convierte cada columna de la matriz formada por dígitos binarios b en un vector
        fila de los valores decimales correspondientes, aplicando codificación de Gray.
    """
    if not torch.is_tensor(b):
        raise Exception('Error: la entrada no es un tensor')
    b = b.long() #Aseguro que sea tipo int
    c = torch.zeros_like(b); c[:,0] = b[:,0]
    for i in range(1,b.size(dim=1)):
        c[:,i] = torch.logical_xor(c[:,i-1], b[:,i])
    c = torch.fliplr(c) # Convierte los bits menos significativos en los más significativos
    #Comprueba un caso especial.
    [n,m] = c.size()
    if torch.min(torch.tensor((m,n))) < 1:
        d = []
        return
    d = c @ 2**torch.arange(m)
    return d 

def t_de2gray(d,n):
    """
    b = de2gray(d,n)
    Convierte un número decimal, d, en un vector binario, b, de longitud n
    """
    c = torch.zeros(len(d),int(n))
    for i in range(int(n)):
        c[:,i] = torch.fmod(d,2) #resto
        d = torch.div(d,2,rounding_mode='floor') #cociente
    c = torch.fliplr(c); b = torch.zeros_like(c); b[:,0] = c[:,0]; aux = b[:,0]
    for i in range(1,int(n)):
        b[:,i] = torch.logical_xor(aux, c[:,i])
        aux = torch.logical_xor(b[:,i], aux) 
    return torch.reshape(b,[-1]).long()

def t_simbolobit(An,alfabeto):
    """
    Bn = simbolobit(An, alfabeto)
    An = secuencia de sí­mbolos pertenecientes al alfabeto
    alfabeto = tabla con los sí­mbolos utilizados en la transmisión 
    Bn = una secuencia de bit, considerando que los sí­mbolos se habí­an
    generado siguiendo una codificación de Gray
    """
    
    k = torch.log2(torch.tensor(len(alfabeto))) # bits por sí­mbolo
    
    if k>1:
        distancia = abs(alfabeto[0]-alfabeto[1])
        indices   = torch.round((An-alfabeto[0])/distancia)
        #Bn        = torch.reshape(t_de2gray(indices,k),int(k*len(An)))
        Bn = t_de2gray(indices,k)
    else:
        Bn = ((An/max(alfabeto))+1)/2
    
    return Bn

def t_detectaSBF(r,alfabeto):
    """
    An = detectaSBF(r,alfabeto)
    r = secuencia a la entrada del detector, con estimaciones de s_i
    alfabeto = tabla con los niveles de amplitud/símbolos  
    
    Genera:
    An = una secuencia de símbolos pertenecientes al alfabeto de acuerdo con
    una regla de distancia euclidiana mínima (mínima distancia)
    """
    
    # Obtiene el índice respecto al alfabeto
    r_repeated = torch.repeat_interleave(r.reshape(r.size(dim=0),1),alfabeto.size(dim=0),1)
    ind = torch.argmin(torch.abs(r_repeated - alfabeto), 1)
    
    # Genera la secuencia de niveles detectados
    An = alfabeto[ind]
    
    return An

def t_transmisorpamV2(Bn,Eb,M,p,L):
    """
    [Xn,Bn,An,phi,alfabetopam] = transmisorpamV2(Bn,Eb,M,p,L) 
    
    Entradas:    
     Bn = Secuencia de dí­gitos binarios
     Eb = Energía media por bit transmitida en Julios
     M  = Número de síímbolos del código PAM
     p  = Pulso paso de baja o paso de banda 
     L  = Número de puntos a utilizar en la representación de un sí­mbolo
    
    Devuelve:
     Xn = la señal de información (discreta)
     Bn = La secuencia de dí­gitos binarios realmente transmitidos
     An = La secuencia de niveles de amplitud transmitidos
   	 phi = Pulso básico real normalizado (energí­a unidad)
     alfabetopam = Los niveles de amplitud asociados a cada sí­mbolo
    """    
    # Paso a tensores de los parámetros de entrada 
    Eb = torch.tensor(Eb)
    M = torch.tensor(M)
    L = torch.tensor(L)
    
    # Comprobación de parámetros de entrada
    p=torch.reshape(p,p.size())  # <=   #p=p.squeeze()
    if Bn.size(dim=1)<1 or Eb<=0 or M<2 or torch.dot(p,p)==0 or L<1:
        raise Exception('Error: revise los parámetros de entrada')
    # Se obtienen en primer lugar los niveles asociado a cada sí­mbolo ¿Cuántos bits hay en cada sí­mbolo?
    k = torch.ceil(torch.log2(M))
    M = 2**(k) # Se Ajusta M a una potencia de dos
    
    # El alfabeto [Ver la ecuación (4.21)] 
    alfabetopam = torch.sqrt(3*Eb*torch.log2(M)/(M**2-1))*(2*(torch.arange(M))-M+1)
    
    # Si Bn no tiene una longitud múltiplo de k, se completa con ceros
    Nb = torch.tensor(Bn.size(dim=1))  # Número de bits a transmitir, actualizado
    Bn = Bn.squeeze().long() #Comprobación de int y dimensiones   
    Bn = torch.hstack((Bn,torch.zeros((k*torch.ceil(Nb/k)-Nb).long()))) #
    Nb = torch.tensor(Bn.size(dim=0))  # Número de bits a transmitir tras la corrección
    Ns = torch.div(Nb,k,rounding_mode='trunc') # Número de sí­mbolos a transmitir
    
    # La secuencia generada
    if M>2:
        An = alfabetopam[t_gray2de(torch.reshape(Bn,(int(Ns),int(k))))]
    else:
        An = alfabetopam[Bn.long()]
    
    # Comprobación de las longitudes y otros datos del pulso suministrado para 
    # hacer que el número de muestras del mismo sea efectivamente L
    Ls = p.size(dim=0)
    if Ls<L:
        p = torch.hstack(p,torch.zeros(L-Ls))
    elif Ls>L:
        print('La duración del pulso se trunca a {} muestras'.format(str(L)))
        p = p[:L] #Debe modificarse si se quiere un pulso de más de L muestras 
    # Se normaliza la energí­a del pulso para obtener la base del sistema
    phi = p / torch.sqrt(p@p)
       
    # Obtención del tren de pulsos, Xn = np.kron(An,phi) ó
    a = torch.reshape(An,(int(Ns),1))*phi
    Xn = torch.reshape(a,(int(Ns)*L,)) #Debe modificarse si se quiere un pulso de más de L muestras
    return [Xn,Bn.long(),An,phi,alfabetopam]

def t_de2Nary(d,n,N):
    """
    b = de2Nary(d,n,N)
    Convierte un número decimal, d, en un vector binario, b, de longitud n
    con base N
    """   
    c = torch.zeros(len(d),int(n))
    for i in range(int(n)):
        c[:,i] = torch.fmod(d,N) #resto
        d = torch.div(d,N,rounding_mode='floor') #cociente
    c = torch.fliplr(c);
    return c.long().T

'''
NEURAL NETWORK
'''

def correction(Nb_train,Nb_test,k,M,n):
    """
    [Nb_train,Ns_train,Nb_test,Ns_test] = correction(Nb_train,Nb_test,k,M,n)
    
    Cambia el valor de Nb_train y Nb_test si es necesario para
    que sea múltiplo de k y M**n
    """
    
    Ns_train = int(torch.floor(torch.tensor(Nb_train/(k*M**n)))) #Number of symbols (multiple of k and M**n)
    Nb_train = int(Ns_train*(k*M**n)) #Number of bits on transmission
    Ns_train = int(torch.floor(torch.tensor(Nb_train/k))) #Number of symbols (multiple of k)

    Ns_test = int(torch.floor(torch.tensor(Nb_test/k))) #Number of symbols (multiple of k)
    Nb_test = int(Ns_test*k)          #Number of bits on transmission

    return [Nb_train,Ns_train,Nb_test,Ns_test]

def create_datasets(H,Nb_train,Nb_test,Eb,M,n,SNRdb,batch_size,valid_size):
    """
    [trainloader,validloader,testloader,x,alphabet] = create_datasets(H,Nb_train,Nb_test,Eb,M,n,SNRdb,batch_size,valid_size)
    
    Devuelve los conjuntos de datos listos para itererar, además del conjunto
    de combinaciones posibles y el alfabeto de la constelación
    """
    # Data generation for training
    [Rn,Cn,Bn,x,alphabet] = generate_data(H,Nb_train,Eb,M,n,SNRdb,train=True)
    trainset = SymbolsDataset(Rn,Cn,Bn)
    
    # Data generation for testing
    [Rn,Cn,Bn,x,alphabet] = generate_data(H,Nb_test,Eb,M,n,SNRdb,train=False)
    testset = SymbolsDataset(Rn,Cn,Bn)
        
    # Indixes used for validation
    num_train = len(trainset)
    split = int(np.floor(valid_size * num_train))
    
    # Split data
    train_split, valid_split = SymbolsDataset.split_data(trainset,split)

    # Load training data in batches
    trainloader = torch.utils.data.DataLoader(train_split,
                                              batch_size=batch_size,
                                              num_workers=0)

    # Load validation data in batches
    validloader = torch.utils.data.DataLoader(valid_split,
                                              batch_size=batch_size,
                                              num_workers=0)
    
    # Load test data in batches
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             num_workers=0)
   
    return [trainloader,validloader,testloader,x,alphabet]

def generate_data(H,Nb,Eb,M,n,SNRdb,train):
    """
    [Xn/Rn,Bn,Cn,x,alphabet] = generate_data(H,Nb,Eb,M,n,SNRdb,test)
    
    Entradas:    
     H     = Matriz del canal
     Nb    = Número de bits a transmitir
     Eb    = Energía media por bit transmitida en Julios
     M     = Número de símbolos del código PAM
     n     = Número de antenas del sistema MIMO nxn
     SNRdb = Signal to Noise Ratio
     test  = Booleano para determinar si son datos para entrenar o testar
    
    Devuelve:
     Rn = La señal de información (discreta) recibida
     Bn = La secuencia de dí­gitos binarios realmente transmitidos
     Cn = La combinación correspondiente a cada valor de An
     x  = La matriz con las posibles combinaciones
     alphabet = Los niveles de amplitud asociados a cada sí­mbolo
    """
    
    k = int(torch.log2(torch.tensor(M)))      #Number of bits on each symbol
    Ns = int(torch.floor(torch.tensor(Nb/k))) #Number of symbols (multiple of k)
    
    alphabet = torch.sqrt(3*Eb*torch.log2(torch.tensor(M))/(M**2-1))*(2*(torch.arange(M))-M+1)
    
    # possible combinations
    ind = t_de2Nary(torch.arange(M**n),n,M) 
    x = alphabet[ind]
    
    if train:  # data for training. same number of symbols transmitted
        
        Xn = x.tile(int(Ns/x.size(1)))
        Xn = Xn[:,torch.randperm(Xn.size(1))] # Shuffle
        Bn = t_simbolobit(Xn.flatten(),alphabet) #Detected bits
        
        #Data reshape. Each row represents an antenna
        Bn = Bn.reshape(n,Nb)
        
        #Index of the combinations
        Cn = getidx(x,Xn,n)
     
    else:     # data for testing
    
        # fixed seed for same results always
        torch.manual_seed(1)
        
        bn = torch.randint(0,2,(1,Nb*n)) #Bits to transmit
        [Xn,Bn,An,phi,alphabet] = t_transmisorpamV2(bn,Eb,M,torch.ones(1),1)
        
        #Data reshape. Each row represents an antenna
        Xn = Xn.reshape(n,Ns)
        Bn = Bn.reshape(n,Nb)
        An = An.reshape(n,Ns)
    
        #Index of the combinations
        Cn = getidx(x,An,n)
        
    SNR = 10**(SNRdb/10)   #Signal to Noise Ratio [n.u.]
    varzn = Eb/(2*SNR)         #Noise variance
    #if varzn <= 0.0025: varzn = torch.tensor(0.0025)
    Wn = torch.sqrt(varzn)*torch.randn(*Xn.shape)  #AWGN
    
    Rn = H@Xn + Wn
    
    return [Rn,Cn,Bn.long(),x,alphabet]

def getidx(x,An,n):
    """
    idx = getidx(x,An,n)
    
    Función que retorna el número de combinación correspondiente a cada
    valor de An
    
    x  = La matriz con las posibles combinaciones
    An = La secuencia de niveles de amplitud transmitidos
    n  = El número de antenas
    """
    
    idx = torch.empty(0)
    for col in An.T:
        i = (sum(x==col.view(n,1)) == n).nonzero(as_tuple=True)[0]
        idx = torch.hstack((idx,i))
    return idx.long()

def train_model(model, trainloader, validloader, optimizer, criterion, patience, n_epochs):
    """
    [model, avg_train_losses, avg_valid_losses] = create_datasets(H,Nb_train,Nb_test,Eb,M,n,SNRdb,batch_size,valid_size):
    
    Función que entrena a la red y retorna el modelo y los errores cometidos

    """
    min_valid_loss = np.inf
    es_counter = 0

    # loss per batch
    train_losses = []
    valid_losses = []
    
    # loss per epoch
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(1, n_epochs + 1):
        
        # Train the model
        model.train()
        for symbols, combs, bits in trainloader:
            # clear the gradients
            optimizer.zero_grad()
            # forward pass
            output = model(symbols)
            # calculate the loss
            loss = criterion(output, combs)
            # backward  pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
            # record training loss 
            train_losses.append(loss.item())
        
        # Validate the model
        model.eval()
        for symbols, combs, bits in validloader:
            # forward pass
            output = model(symbols)
            # calculate the loss
            loss = criterion(output, combs)
            # record valid loss 
            valid_losses.append(loss.item())
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss) 

        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')
        else:
            es_counter += 1
            if es_counter == patience: break
            
        # load saved model
        model.load_state_dict(torch.load('saved_model.pth'))
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
    return [model, avg_train_losses, avg_valid_losses]

def printloss(avg_train_losses,avg_valid_losses,SNR,save):
    """
    Función para mostrar por consola las pérdidas de entramiento y validación
    en un epoch
    """

    fig = plt.figure(1,figsize=(10,8))
    plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
    plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2) # VARIABLE scale
    plt.xlim(0, len(avg_valid_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title('SNR = {}dB'.format(SNR))
    plt.tight_layout()
    plt.show()
    if save: fig.savefig('loss_plot.png', bbox_inches='tight')
    pass

def eval_model(model, trainloader, validloader, testloader, x, alphabet, k):
    """
    [berTrain, berValid, berTest] = eval_model(model, trainloader, validloader, testloader, x, alphabet, k)
    
    Función que evalua el modelo y devuelve las distintas tasas de error binarias
    """
    
    # BER for saved model with train data
    model.eval()
    
    berTrain = 0
    for symbols, combs, bits in trainloader: 
        # forward pass
        output = model(symbols)
        berTrain += ber(output,x,alphabet,bits,k)  
    berTrain /= len(trainloader)
    
    berValid = 0
    for symbols, combs, bits in validloader:  
        # forward pass
        output = model(symbols)
        berValid += ber(output,x,alphabet,bits,k)  
    berValid /= len(validloader)
    
    berTest = 0
    for symbols, combs, bits in testloader:  
        # forward pass
        output = model(symbols)
        berTest += ber(output,x,alphabet,bits,k)  
    berTest /= len(testloader)

    return [berTrain, berValid, berTest]

def ber(output,x,alphabet,bits,k):
    """
    ber = ber(output,x,alphabet,bits,k)
    Calcula la tasa de error de bit de una señal transmitida
    
    Entradas:
     output   = Salida de la Red Neuronal (bacth_size x n_combinations)
     x        = La matriz con las posibles combinaciones
     alphabet = Los niveles de amplitud asociados a cada sí­mbolo
     bits     = El conjunto de Bn transmitidos
     k        = Los bits que hay por símbolo
    """
    
    Nb = k*output.size(0) # Número de bits transmitidos (k*Ns)
    n = x.size(0)         # Núermo de antenas

    val, idx = torch.max(output,1) # El máximo de cada salida
    
    # Es necesario modificar la variable 'bits' de manera que estén
    # en la forma nxNb, correspondiendo cada fila a la transmisión de una antena
    Bn = torch.empty(0,Nb)
    for i in range(int(n)): Bn = torch.vstack((Bn,bits[:,i,:].flatten()))
    
    
    Andetected = x[:,idx]    # Detected symbols
    Bndetected = t_simbolobit(Andetected.flatten(),alphabet).reshape(n,Nb) #Detected bits
    errorsBit = Nb-torch.sum(Bndetected==Bn,axis=1)

    return (errorsBit/Nb).reshape(n,1)

def detect(Rn,Bn,H,alphabet,Nb,Ns,n,method,**kwargs):
    """
    ber = detect(Rn,Bn,H,alphabet,Nb,Ns,n,method,**kwargs)
    
    Calcula la BER de los métodos ZF, LMMSE y ML
    """
    if method == 'ZF':
        Hinv = kwargs.get('Hinv',None)
        
        Z = Hinv@Rn
        
    elif method == 'LMMSE':
        Es = kwargs.get('Es',None)
        varzn = kwargs.get('varzn',None)
        
        Z = torch.linalg.inv(H.T@H + varzn/Es*torch.eye(n))@H.T@Rn
        
    elif method == 'ML':
        x = kwargs.get('x',None)

        hx = torch.unsqueeze(H@x,2) #H@x with shape (n,combinations,1)
        Z = x[:,torch.argmin(torch.linalg.norm(Rn.reshape(n,1,Ns)-hx,dim=0),axis=0)]
    
    Andetected = t_detectaSBF(Z.flatten(),alphabet).reshape(n,Ns) #Detected symbols
    Bndetected = t_simbolobit(Andetected.flatten(),alphabet).reshape(n,Nb) #Detected bits
    errorsBit = Nb-torch.sum(Bndetected==Bn,axis=1) # Nº Errors per bit
    
    return (errorsBit/Nb).reshape(n,1)