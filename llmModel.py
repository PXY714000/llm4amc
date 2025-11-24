import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import transformers
#from transformers import GPT2ForSequenceClassification
#from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoModelForCausalLM 
from einops import rearrange
from dataEmbedding import dataEmbedding



#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class seBlock(nn.Module):
    def __init__(self, inputPlanes,ratio = 4):
        super(seBlock,self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(inputPlanes,inputPlanes//ratio,1,bias = False)
        self.relu1 = nn.ReLU() 
        self.fc2 = nn.Conv2d(inputPlanes//ratio,inputPlanes,1,bias = False)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        #print(x.shape)
        avgOutput = self.fc2(self.relu1(self.fc1(self.avgPool(x))))
        maxOutput = self.fc2(self.relu1(self.fc1(self.maxPool(x))))
        output = avgOutput + maxOutput
        #print(output.shape)
        output = self.sigmoid(output)
        return output

class coreResidualBlock(nn.Module):
    def __init__(self, inputPlanes):
        super(coreResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(inputPlanes,inputPlanes,3,1,1)
        self.conv2 = nn.Conv2d(inputPlanes,inputPlanes,3,1,1)
        self.relu = nn.ReLU(inplace=True)
        self.se = seBlock(inputPlanes = inputPlanes,ratio = 1)
        
    def forward(self,x):
        #print(x.shape)
        result1 = self.conv1(x)
        result1 = self.relu(result1)
        result1 = self.conv2(result1)

        #print(result1.shape)
        result2 = self.se(result1)
        #print(result2.shape)
        
        output = result2 * result1

        finalResult= torch.add(x,output)
        return finalResult

class llmModel(nn.Module):

    def __init__(self, patchSize = 4, residualLayerNumber = 4, residualDimensionNumber = 64, 
                 gptLayerNumber = 6 , mlp = 0, dropout = 0.1,
                 historyLength = 16, predictionLength= 1,rbNumber = 48, 
                 gpuId = 0, whetherUseGpu = 1):
        super(llmModel,self).__init__()

        self.device = torch.device('cuda:{}'.format(gpuId))
        self.patchSize = patchSize

        self.patchLayer = nn.Linear(self.patchSize,self.patchSize)
        #self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        #self.gpt2.h = self.gpt2.h[:gptLayerNumber]

        #self.gptDimensionNumber = 768

        self.qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True, 
            #output_attentions=True,
            #output_hidden_states=True,
            #device_map="auto",
            #device_map="None",
            torch_dtype=torch.float32 # 
        )
        #print(self.qwen)  # 
        #print(self.qwen.model)  # 
        #print(self.qwen.model.layers)  # 
        self.qwen.model.layers = self.qwen.model.layers[:gptLayerNumber] # 
        self.gptDimensionNumber = 896 

        for name, param in self.qwen.named_parameters():
            if 'norm' in name:  # 
                param.requires_grad = True
            else:
                param.requires_grad = False

        if whetherUseGpu:
            device = torch.device('cuda:{}'.format(gpuId))
            self.qwen.to(device=device)

        self.residualBlock = nn.Sequential(nn.Conv2d(1,residualDimensionNumber,3,1,1))
        for idx in range(residualLayerNumber):
            self.residualBlock.append(coreResidualBlock(residualDimensionNumber))
        self.residualBlock.append(nn.Conv2d(residualDimensionNumber,1,3,1,1))

        self.encodingEmbedding = dataEmbedding(c_in = rbNumber, d_model = self.gptDimensionNumber, dropout=dropout)

        self.preprocessingLayer = nn.Linear(historyLength,historyLength)

        self.outputDimensionLayer = nn.Linear(self.gptDimensionNumber, rbNumber)
        self.outputLengthLayer = nn.Linear(historyLength,predictionLength)
        self.sigmoid = nn.Sigmoid()
        #self.outputRelu = nn.ReLU(inplace=True)
    
    def forward(self,xEncoding,xMarkEncoding,xMarkDecoding,mask = None):
        #norm(1024 16 48)
        meanofInput = torch.mean(xEncoding)
        stdofInput = torch.std(xEncoding)
        xEncoding = (xEncoding-meanofInput) / stdofInput
        #print(xEncoding.shape)

        batchSize, historyLength, rbNumber = xEncoding.shape

        #patching(1024 16 48)
        xEncoding = xEncoding.reshape(batchSize,historyLength//self.patchSize,self.patchSize,rbNumber)
        #print(xEncoding.shape)
        xEncoding = self.patchLayer(xEncoding.permute(0,1,3,2)).permute(0,1,3,2)
        #print(xEncoding.shape)
        xEncoding = xEncoding.reshape(batchSize,historyLength,rbNumber)
        #print(xEncoding.shape)

        #sinrAttention(1024 16 48)
        xEncoding = rearrange(xEncoding,'b l (k o) -> b o l k', o =1) #(1024 1 16 48)
        #print(xEncoding.shape)
        xEncoding = self.residualBlock(xEncoding)
        #print(xEncoding.shape)
        xEncoding = rearrange(xEncoding,'b o l k -> b l (k o)') #(1024 16 48)
        #print(xEncoding.shape)

        #Embedding(1024 16 768)
        encodingOut = self.encodingEmbedding(xEncoding,xMarkEncoding)
        #print(encodingOut.shape)

        #FC(1024 16 768)
        encodingOut = self.preprocessingLayer(encodingOut.permute(0,2,1)).permute(0,2,1)
        #print(encodingOut.shape)
        encodingOut = torch.nn.functional.pad(encodingOut,(0,self.gptDimensionNumber - encodingOut.shape[-1]))
        #print(encodingOut.shape)

        #gpt2(1024 16 768)
        attention_mask = torch.ones(encodingOut.shape[:2], dtype=torch.long, device=encodingOut.device)
        outputs = self.qwen(
            inputs_embeds=encodingOut,
            attention_mask=attention_mask,
            return_dict=True
        )
        decodingOut = outputs[0]
        #print(decodingOut.shape)
        decodingOut = decodingOut[:,:,:self.gptDimensionNumber]
        #print(decodingOut.shape)


        #output(1024 1 48)
        decodingOut = self.outputDimensionLayer(decodingOut)
        #print(decodingOut.shape)
        decodingOut = self.outputLengthLayer(decodingOut.permute(0,2,1)).permute(0,2,1)
        #print(decodingOut.shape)

        #disnorm(1024 1 48)
        decodingOut = decodingOut * stdofInput + meanofInput
        #print(decodingOut.shape)

        #decodingOut = self.sigmoid(decodingOut)

        return decodingOut
    

if __name__ == '__main__':
    import torch
    print(transformers.__version__)
    print(os.environ.get('HF_ENDPOINT'))

    device = torch.device('cuda:0')
    model = llmModel().to(device)
    inputs = torch.rand(1, 16, 48).to(device)
    #inputs = inputs.half()
    startTime = time.time()
    out = model(inputs, None, None, None)
    inferenceTime = time.time()-startTime
    print(f"Inference time: {inferenceTime:.4f} seconds")
    print(out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
