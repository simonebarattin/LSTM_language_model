import torch
import torch.nn as nn

'''
    Script with the implementation of a GLU block from [1]. The code is an adaptation of [2]

    Args:
        kernel_size (int)  : size of the filter used in CNN
        in_channels (int)  : input channels
        out_channels (int) : output channels
        bottleneck (int)   : value used to reduce the size of input and output channels to create a bottleneck

    Output:
        H + residual (torch.FloatTensor) : output of the Gated CNN block + residual

    References:
        [1] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language modeling with gated convolutional networks,” 2016
        [2] https://github.com/DavidWBressler/GCNN/blob/master/GCNN.ipynb
'''
class GLUblock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, bottleneck):
        super(GLUblock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.bottleneck = bottleneck

        self.project = False if in_channels == out_channels else True
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.residual = nn.utils.weight_norm(self.residual)
        
        self.leftpad = nn.ConstantPad2d((0, 0, kernel_size-1, 0), 0) # add 0 padding to the left of the sequence to obtain causal convolution 

        # Convolution block A with bottleneck
        self.convA_1 = nn.Conv2d(in_channels, int(in_channels/bottleneck), kernel_size=(1, 1))
        self.convA_1 = nn.utils.weight_norm(self.convA_1, name='weight', dim=0)
        self.convA_2 = nn.Conv2d(int(in_channels/bottleneck), int(in_channels/bottleneck), kernel_size=(kernel_size, 1))
        self.convA_2 = nn.utils.weight_norm(self.convA_2, name='weight', dim=0)
        self.convA_3 = nn.Conv2d(int(in_channels/bottleneck), out_channels, kernel_size=(1, 1))
        self.convA_3 = nn.utils.weight_norm(self.convA_3, name='weight', dim=0)
        self.convA = nn.Sequential(
            self.convA_1,
            self.convA_2,
            self.convA_3
        )

        # Gating convolutional block B with bottleneck
        self.convB_1 = nn.Conv2d(in_channels, int(in_channels/bottleneck), kernel_size=(1, 1))
        self.convB_1 = nn.utils.weight_norm(self.convB_1, name='weight', dim=0)
        self.convB_2 = nn.Conv2d(int(in_channels/bottleneck), int(in_channels/bottleneck), kernel_size=(kernel_size, 1))
        self.convB_2 = nn.utils.weight_norm(self.convB_2, name='weight', dim=0)
        self.convB_3 = nn.Conv2d(int(in_channels/bottleneck), out_channels, kernel_size=(1, 1))
        self.convB_3 = nn.utils.weight_norm(self.convB_3, name='weight', dim=0)
        self.convB = nn.Sequential(
            self.convB_1,
            self.convB_2,
            self.convB_3
        )

    def forward(self, x):
        residual = x
        if self.project: # if input channels != output channels, change size of residual, otherwise we cannot add it
            residual = self.residual(residual)

        x = self.leftpad(x) # [bs,in_c,seq_length+(k-1),1]
        A = self.convA(x)
        B = self.convB(x)
        H = torch.mul(A, torch.sigmoid(B)) # [bs,out_c,seq_length,1]
        
        return H + residual

'''
    Script with implementation of the Gated CNN from [1].

    Args:
        vocab_size (int)     : size of the dataset vocabulary
        embedding_size (int) : dimensionality of the embedding (defualt=400)
        kernel_size (int)    : size of the filter used in CNN
        out_channels (int)   : output channels
        num_layers (int)     : number of stacked GLU blocks
        bottleneck (int)     : value used to reduce the size of input and output channels to create a bottleneck
    
    Output:
        output (torch.FloatTensor) : target log probabilities for each example
        loss (tensor)              : computed negative log likelihood loss

    Reference:
        [1] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language modeling with gated convolutional networks,” 2016
'''
class CNN_LM(nn.Module):
    def __init__(self, vocab_size, embedding_size, kernel_size, out_channels, num_layers, bottleneck):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_size
        self.kernel = kernel_size
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bottleneck = bottleneck
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.no_res_GLU = GLUblock(kernel_size, embedding_size, out_channels, bottleneck)
        self.res_GLUs = self.make_GLU_layers(kernel_size, out_channels, num_layers, bottleneck)
        self.output = nn.AdaptiveLogSoftmaxWithLoss(out_channels, vocab_size, cutoffs=[round(vocab_size/25),round(vocab_size/5)],div_value=4)

    def make_GLU_layers(self, kernel_size, out_channels, num_layers, bottleneck):
        layers = [GLUblock(kernel_size, out_channels, out_channels, bottleneck) for i in range(num_layers)]
        return nn.Sequential(*layers)
        
    def forward(self, x, y):
        x = x.permute(1, 0)
        y = y.reshape(-1)

        emb = self.embedding(x)
        emb = emb.transpose(2, 1)
        emb = emb.unsqueeze(3)

        conv1 = self.no_res_GLU(emb)
        output = self.res_GLUs(conv1)

        output = output.squeeze(3)
        output = output.contiguous().view(-1, self.out_channels)
        output, loss = self.output(output, y) # AdaptiveLogSoftmaxWithLoss uses negative log likelihood loss (same as CE) so it's fine
        return output, loss