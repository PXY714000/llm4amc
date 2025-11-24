import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random

def dBm2W(power_dBm):
    power_W = 10 ** ((power_dBm-30)/10)
    return power_W

def linear2dB(linearValue):
    if torch.any(linearValue <= 0):
        raise ValueError("输入的线性值必须为正数")
    dBValue = 10 * torch.log10(linearValue)
    return dBValue

def dB2linear(dBvalue):
    linaerValue = 10 ** (dBvalue / 10)
    return linaerValue    

def csi2Sinr(csi):
    powerofBS_dBm = 1
    powerofNoise_dBm = -15
    powerofBS_W = dBm2W(powerofBS_dBm)
    powerofNoise_W = dBm2W(powerofNoise_dBm)

    powerofCsi_W = np.abs(csi) ** 2

    sinr = powerofBS_W * powerofCsi_W / powerofNoise_W
    sinr = np.abs(csi) ** 2
    return sinr

def addNoise(sinr,snr):
    signalPower = np.mean(sinr**2)
    noisePower = signalPower / (10**(snr/10))
    noise = np.sqrt(noisePower) * np.random.randn(*sinr.shape)
    sinrwithNoise = sinr + noise
    return sinrwithNoise

class datasetProcessing(data.Dataset):
    def __init__(self,historyDataPath, predictionDataPath, whetherTraining = 1, whetherFewshot = 0,
                 trainingsetPercentage = 0.9, validationPercentage = 0.1):
        super(datasetProcessing).__init__()
        
        #read data(900,320,16,48),(900,320,1,48)
        historyCsi = hdf5storage.loadmat(historyDataPath)['H_U_his']
        predictionCsi = hdf5storage.loadmat(predictionDataPath)['H_U_pre']

        #print(historyCsi.shape)
        #print(predictionCsi[0,0,:])
        #historyCsi = historyCsi[4:6,...]
        #predictionCsi = predictionCsi[4:6,...]

        historySinr = csi2Sinr(historyCsi)
        predictionSinr = csi2Sinr(predictionCsi)
        print(historySinr.shape)
        print(predictionSinr.shape)
        #print(historySinr.shape)
        #print(predictionSinr.shape)
        #print(predictionSinr[0,0,:])

        #distinguish trainingset and validation set(900,288,16,48),(900,288,1,48)
        batchNumber = predictionSinr.shape[1]

        if whetherTraining:
            historySinr = historySinr[:,:int(trainingsetPercentage * batchNumber), ...]
            predictionSinr = predictionSinr[:,:int(trainingsetPercentage*batchNumber),...]
        else:
            historySinr = historySinr[:,int(trainingsetPercentage * batchNumber):int((trainingsetPercentage+validationPercentage) * batchNumber), ...]
            predictionSinr = predictionSinr[:,int(trainingsetPercentage*batchNumber):int((trainingsetPercentage+validationPercentage) * batchNumber),...]

        #print(historySinr.shape)
        #print(predictionSinr.shape)
        #print(predictionSinr[0,0,:])

        #(25920,16,48)(25920,1,48)
        historySinr = rearrange(historySinr,'v b l k -> (v b) l k')
        predictionSinr = rearrange(predictionSinr,'v b l k -> (v b) l k')
        #print(historySinr.shape)
        #print(predictionSinr.shape)
        #print(predictionSinr[0,0,:])

        #(25920,16,48)(25920,1,48)
        historyLength = historySinr.shape[1]
        predictionLength = predictionSinr.shape[1]
        
        #(25920,17,48)
        allData = np.concatenate((historySinr,predictionSinr), axis = 1)
        #print(allData.shape)
        #(25920,17,48)
        np.random.shuffle(allData)
        #print(allData.shape)

        historySinr = allData[:,:historyLength,...]
        predictionSinr = allData[:,-predictionLength:,...]

        #add noise
        sampleNumber = historySinr.shape[0]
        for idx in range(sampleNumber):
           historySinr[idx,...] = addNoise(historySinr[idx,...],15+random.rand() * 10)
           #predictionSinr[idx,...] = addNoise(predictionSinr[idx,...],10+random.rand() * 15)

        #print(historySinr.shape)
        #print(predictionSinr.shape)
        #print(historySinr[0,0,0])
        #print(predictionSinr[0,0,0])
        #nomalization
        meanofHistorySinr = np.mean(historySinr)
        stdofHistorySinr = np.std(historySinr)
        maxofHistorySinr = np.max(historySinr)
        minofHistorySinr = np.min(historySinr)
        #print(maxofHistorySinr)
        #print(minofHistorySinr)

        historySinr = (historySinr - minofHistorySinr)/(maxofHistorySinr- minofHistorySinr)
        predictionSinr = (predictionSinr - minofHistorySinr)/(maxofHistorySinr - minofHistorySinr)
        #print(historySinr[0,0,0])
        #print(predictionSinr[0,0,0])

        #historySinr = (historySinr - meanofHistorySinr) / stdofHistorySinr
        #predictionSinr = (predictionSinr - meanofHistorySinr) / stdofHistorySinr
        #print(historySinr.shape)
        #print(predictionSinr.shape)
        #print(predictionSinr[0,0,:])

        #few_shot(2592,16,48)(2592,1,48)
        if whetherFewshot == 1:
            historySinr = historySinr[::10,...]
            predictionSinr = predictionSinr[::10,...]

        historySinr = torch.tensor(historySinr, dtype=torch.float32)
        predictionSinr = torch.tensor(predictionSinr,dtype=torch.float32)
        print(historySinr.shape)
        print(predictionSinr.shape)
        #print(predictionSinr[0,0,0])
        self.historySinr = historySinr
        self.predictionSinr = predictionSinr


    def __getitem__(self, index):
        return self.predictionSinr[index,:].float(),\
               self.historySinr[index,:].float()
    
    def __len__(self):
        return self.predictionSinr.shape[0]