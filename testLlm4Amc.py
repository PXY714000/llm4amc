import time
import torch
import numpy as np
from einops import rearrange
import hdf5storage
import tqdm

from lossFunction import nmseLoss,mseLoss
from dataset import dBm2W, csi2Sinr,addNoise

if __name__ == "__main__":

    device = torch.device('cuda:0')
    
    testHistoryDataPath = "./1,1_train_10/H_U_his.mat"
    testPredictionDataPath = "./1,1_train_10/H_U_pre.mat"

    modelPath ="./weights/llm4Amc-1.1(10)_noise(15-25).pth"

    historyLength = 16
    predictionLength = 1

    criterion1 = nmseLoss()
    criterion2 = mseLoss()

    #read data
    testHistoryCsi = hdf5storage.loadmat(testHistoryDataPath)['H_U_his']
    testPredictionCsi = hdf5storage.loadmat(testPredictionDataPath)['H_U_pre']
    
    historySinr = csi2Sinr(testHistoryCsi)
    predictionSinr = csi2Sinr(testPredictionCsi)

    model = torch.load(modelPath,map_location=device).to(device)

    print("---------------------------------------------------------------------")

    for noiseIdx in range(0,7):

        for velocityIdx in range(4,5):
            testLossStackLlm = []
            testLossStackNp = []
        
            #read data in every velocity (1,1000,16,48)(1,1000,1,48)
            historySinrEachVelocity = historySinr[[velocityIdx],...]
            predictionSinrEachVelocity = predictionSinr[[velocityIdx],...]

            #rearranging (1000,16,48)(1000,1,48)
            historySinrEachVelocity = rearrange(historySinrEachVelocity,'v b l k -> (v b) l k')
            predictionSinrEachVelocity = rearrange(predictionSinrEachVelocity,'v b l k -> (v b) l k')

            #nomalization(1000,16,48)(1000,1,48)
            #meanofHistorySinr = np.mean(historySinrEachVelocity)
            #stdofHistorySinr = np.std(historySinrEachVelocity)
            #historySinrEachVelocity = (historySinrEachVelocity - meanofHistorySinr) / stdofHistorySinr
            #predictionSinrEachVelocity = (predictionSinrEachVelocity - meanofHistorySinr) /stdofHsistorySinr
            if noiseIdx != 0:
                noiseSNR = 5*noiseIdx-5
                print('SNR:',noiseSNR)
                historySinrEachVelocity = addNoise(historySinrEachVelocity,noiseSNR)
            #predictionSinrEachVelocity = addNoise(predictionSinrEachVelocity,25)
        

            maxnofHistorySinr = np.max(historySinrEachVelocity)
            minofHistorySinr = np.min(historySinrEachVelocity)
            #denom = maxnofHistorySinr - minofHistorySinr
            #if denom == 0:
             #   denom = 1e-8
            #historySinrEachVelocity = (historySinrEachVelocity - minofHistorySinr) / denom
            #predictionSinrEachVelocity = (predictionSinrEachVelocity - minofHistorySinr) /denom
            historySinrEachVelocity = (historySinrEachVelocity - minofHistorySinr) / (maxnofHistorySinr-minofHistorySinr)
            predictionSinrEachVelocity = (predictionSinrEachVelocity - minofHistorySinr) /(maxnofHistorySinr-minofHistorySinr)


            historySinrEachVelocity = torch.tensor(historySinrEachVelocity, dtype=torch.float32)
            predictionSinrEachVelocity = torch.tensor(predictionSinrEachVelocity,dtype=torch.float32)
        
            #evaluation
            model.eval()
            bs = 64
            lens = historySinrEachVelocity.shape[0]
            cycle_times = lens //bs
            with torch.no_grad():
                for cyt in range(cycle_times):
                    historySinrEachVelocity = historySinrEachVelocity[cyt*bs:(cyt+1)*bs,:,:].to(device)
                    predictionSinrEachVelocity = predictionSinrEachVelocity[cyt*bs:(cyt+1)*bs,:,:].to(device)
                    predictionSinrEachVelocityLlm_hat = model(historySinrEachVelocity,None,None,None)
                    predictionSinrEachVelocityNp_hat = historySinrEachVelocity[:,[-1],:]

                    #predictionSinrEachVelocityNp_hat = predictionSinrEachVelocityNp_hat*(maxnofHistorySinr-minofHistorySinr)+minofHistorySinr
                    #predictionSinrEachVelocityLlm_hat = predictionSinrEachVelocityLlm_hat*(maxnofHistorySinr-minofHistorySinr)+minofHistorySinr

                    lossNp = criterion1(predictionSinrEachVelocityNp_hat,predictionSinrEachVelocity)
                    lossLlm = criterion1(predictionSinrEachVelocityLlm_hat,predictionSinrEachVelocity)

                    testLossStackLlm.append(lossLlm.item())
                    testLossStackNp.append(lossNp.item())

            print("velocity", velocityIdx, ":  NMSE_Llm:",  np.nanmean(np.array(testLossStackLlm)))
            print("velocity", velocityIdx, ":  NMSE_Np:", np.nanmean(np.array(testLossStackNp)))
        


        








