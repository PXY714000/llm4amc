import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio

import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter

from lossFunction import nmseLoss
from llmModel import llmModel
from dataset import datasetProcessing


# ============= HYPER PARAMS(Pre-Defined) ==========#
learningRate = 0.0001
traingEpochs = 300
batchSize = 256
device = torch.device('cuda:0')

currentLoss = 100
savingPath = "weights/llm4Amc-Qwen2.5-0.5B-Instruct.pth"
trainingHistoryDataPath = "./data/train/H_his.mat"
trainingPredictionDataPath = "./data/train/H_pre.mat"

#creat trainingSet and validationSet
trainingSet = datasetProcessing(trainingHistoryDataPath,trainingPredictionDataPath, whetherTraining = 1, whetherFewshot = 0)  # creat data for training
validationSet = datasetProcessing(trainingHistoryDataPath,trainingPredictionDataPath, whetherTraining = 0, whetherFewshot = 0)  # creat data for validation

model = llmModel().to(device)

if os.path.exists(savingPath):
    model = torch.load(savingPath, map_location=device)

def savingBestCheckpoint(model):  # save model function
    modelSavingPath = savingPath
    torch.save(model, modelSavingPath)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def train(trainingDataLoader, validationDataLoader):
    global traingEpochs, currentLoss
    print('Start training...')
    for epoch in range(traingEpochs):
        epochTrainingLoss, epochValidationLoss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(trainingDataLoader, 1):
            predictionSinr, historySinr = Variable(batch[0]).to(device), \
                           Variable(batch[1]).to(device)
            optimizer.zero_grad()  # fixed
            predictionSinr_hat = model(historySinr, None, None, None)
            loss = criterion(predictionSinr_hat, predictionSinr)  # compute loss
            epochTrainingLoss.append(loss.item())  # save all losses into a vector for one epoch
            loss.backward()
            optimizer.step()

        #       lr_scheduler.step()  # update lr

        totalTrainingLoss = np.nanmean(np.array(epochTrainingLoss))  # compute the mean value of all losses, as one epoch loss
        print('Epoch: {}/{} training loss: {:.7f}'.format(epoch+1, traingEpochs, totalTrainingLoss))  # print loss for each epoch

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validationDataLoader, 1):
                predictionSinr, historySinr = Variable(batch[0]).to(device), \
                               Variable(batch[1]).to(device)
                optimizer.zero_grad()  # fixed
                predictionSinr_hat = model(historySinr, None, None, None)
                loss = criterion(predictionSinr_hat, predictionSinr)  # compute loss
                epochValidationLoss.append(loss.item())  # save all losses into a vector for one epoch
            totalValidationLoss = np.nanmean(np.array(epochValidationLoss))
            print('validate loss: {:.7f}'.format(totalValidationLoss))
            if totalValidationLoss < currentLoss:
                currentLoss = totalValidationLoss
                savingBestCheckpoint(model)


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    trainingDataLoader = DataLoader(dataset=trainingSet, num_workers=0, batch_size=batchSize, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    validationDataLoader = DataLoader(dataset=validationSet, num_workers=0, batch_size=batchSize,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    optimizer = optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), weight_decay=0.0001)

    criterion = nmseLoss().to(device)
    train(trainingDataLoader, validationDataLoader)  # call train function (

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

