import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/DavidWBressler/GCNN/blob/master/GCNN.ipynb
class GLUblock(nn.Module):
    def __init__(self, k, in_c, out_c, downbot):
        super().__init__()
        #only need to change shape of the residual if num_channels changes (i.e. in_c != out_c)
        #[bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]
        if in_c == out_c:
            self.use_proj=0
        else:
            self.use_proj=1
        self.convresid=nn.utils.weight_norm(nn.Conv2d(in_c, out_c, kernel_size=(1,1)),name='weight',dim=0)
        
        self.leftpad = nn.ConstantPad2d((0,0,k-1,0),0)#(paddingLeft, paddingRight, paddingTop, paddingBottom)

        #[bs,in_c,seq_length+(k-1)]->conv(1,in_c,in_c/downbot)->[bs,in_c/downbot,seq_length+(k-1)]
        self.convx1a = nn.utils.weight_norm(nn.Conv2d(in_c, int(in_c/downbot), kernel_size=(1,1)),name='weight',dim=0)
        self.convx2a = nn.utils.weight_norm(nn.Conv2d(in_c, int(in_c/downbot), kernel_size=(1,1)),name='weight',dim=0)
        #[bs,in_c/downbot,seq_length+(k-1)]->conv(k,in_c/downbot,in_c/downbot)->[bs,in_c/downbot,seq_length]
        self.convx1b = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), int(in_c/downbot), kernel_size=(k,1)),name='weight',dim=0)
        self.convx2b = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), int(in_c/downbot), kernel_size=(k,1)),name='weight',dim=0)
        #[bs,in_c/downbot,seq_length]->conv(1,in_c/downbot,out_c)->[bs,out_c,seq_length]
        self.convx1c = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), out_c, kernel_size=(1,1)),name='weight',dim=0)
        self.convx2c = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), out_c, kernel_size=(1,1)),name='weight',dim=0)

    def forward(self, x):
        residual = x
        if self.use_proj==1:# if in_c != out_c, need to change size of residual
            residual=self.convresid(residual)
        x=self.leftpad(x) # [bs,in_c,seq_length+(k-1),1]
        x1 = self.convx1c(self.convx1b(self.convx1a(x))) # [bs,out_c,seq_length,1]
        x2 = self.convx2c(self.convx2b(self.convx2a(x))) # [bs,out_c,seq_length,1]
        x2 = torch.sigmoid(x2)
        x=torch.mul(x1,x2) # [bs,out_c,seq_length,1]
        return x+residual

class CNN_LM(nn.Module):
    def __init__(self, vs, emb_sz, k, nh, nl,downbot):
    #def __init__(self, vs, emb_sz, k, nh, nl,dw,cutoffs):
        super().__init__()
        
        self.embed = nn.Embedding(vs, emb_sz)
        
        self.inlayer=GLUblock(k,emb_sz,nh,downbot)
        self.GLUlayers=self.make_GLU_layers(k,nh,nl,downbot)
        self.out=nn.AdaptiveLogSoftmaxWithLoss(nh, vs, cutoffs=[round(vs/25),round(vs/5)],div_value=4)

    def make_GLU_layers(self, k, nh, nl, downbot):
        layers = [GLUblock(k, nh, nh, downbot) for i in range(nl)]
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        target=x[1:,:]
        target=target.contiguous().view(target.size()[0]*target.size()[1])#[seq_length*bs,out_c]
        x=x[:-1,:]
        
        #first block
        x = self.embed(torch.t(x)) # x -> [seq_length,bs] -> [bs,seq_length] -> [bs,seq_length,emb_sz] ... i.e. transpose 1st
        x=torch.transpose(x, 1, 2) #[bs,emb_sz,seq_length]    
        x = x.unsqueeze(3)  # [bs,emb_sz,seq_length,1]
        x=self.inlayer(x) #[bs,nh,seq_length,1]
             
        #residual GLU blocks
        x=self.GLUlayers(x) # [bs,nh,seq_length,1]
        
        #out
        x=torch.squeeze(x,3) #[bs,out_c,seq_length]
        x=torch.transpose(x, 1, 2) #[bs,seq_length,out_c]
        x=torch.transpose(x, 0, 1) #[seq_length,bs,out_c]
        x=x.contiguous().view(-1,x.size()[2])#[seq_length*bs,out_c]
        outta=self.out(x,target)
        
        return    outta