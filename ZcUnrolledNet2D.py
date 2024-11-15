import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import ZcEncoderDecoder2p1D_New as UNet
import ZcDC_2D as DC

class ZcUnrolledNet(nn.Module):
    
    def __init__(self, miu):
        super().__init__()
        '''
        # ResNet:
        self.Network_M1 = ResNet.ZcResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)  # Unet taking 2 channels output 2 channels
        self.Network_M2 = ResNet.ZcResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)  # Unet taking 2 channels output 2 channels
        self.Network_M3 = ResNet.ZcResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)  # Unet taking 2 channels output 2 channels
        self.Network_M4 = ResNet.ZcResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)  # Unet taking 2 channels output 2 channels
        
        self.Network_R4_P1 = ResNet.ZcResNet(input_channels = 2, intermediateChannels = 64, output_channels = 2)  # Unet taking 2 channels output 2 channels
        '''
        # Unet:
        self.Network_R24_1 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R24_2 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)

        self.Network_R20_1 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R20_2 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)

        self.Network_R16_1 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R16_2 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)

        self.Network_R12_1 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R12_2 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)

        self.Network_R8_1 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R8_2 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)

        self.Network_R4_1 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R4_2 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R4_3 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R4_4 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)
        self.Network_R4_5 = UNet.PromptUnet(in_chans = 2*3*3, out_chans = 2)

        # it takes and output real-valued image, but it convert between R2C and C2R within it. 

        self.DataConsistency_R24_1 = DC.ZcDC_2D(miu)
        self.DataConsistency_R24_2 = DC.ZcDC_2D(miu)

        self.DataConsistency_R20_1 = DC.ZcDC_2D(miu)
        self.DataConsistency_R20_2 = DC.ZcDC_2D(miu)

        self.DataConsistency_R16_1 = DC.ZcDC_2D(miu)
        self.DataConsistency_R16_2 = DC.ZcDC_2D(miu)

        self.DataConsistency_R12_1 = DC.ZcDC_2D(miu)
        self.DataConsistency_R12_2 = DC.ZcDC_2D(miu)

        self.DataConsistency_R8_1 = DC.ZcDC_2D(miu)
        self.DataConsistency_R8_2 = DC.ZcDC_2D(miu)

        self.DataConsistency_R4_1 = DC.ZcDC_2D(miu)
        self.DataConsistency_R4_2 = DC.ZcDC_2D(miu)
        self.DataConsistency_R4_3 = DC.ZcDC_2D(miu)
        self.DataConsistency_R4_4 = DC.ZcDC_2D(miu)
        self.DataConsistency_R4_5 = DC.ZcDC_2D(miu)


    def UnrolledBlock(self, recon, zf, coil, mask, maskEmb, network, dc):
        # recon, zerofilled, coil, mask, psf, CNN, DC
        recon = checkpoint(network, recon, maskEmb)
        recon = checkpoint(dc, recon, zf, coil, mask)
        
        return recon
    
    
    # here the forward step of unrolled network. 
    # it takes real-valued zero-filled image as input
    def forward(self, zerofilled, coil, mask, maskEmb, rate):
        
        # recon starts from zero-filled image
        recon =  nn.Parameter(torch.clone(zerofilled), requires_grad=True)
        # then it goes through DC and CNN alternatively 
        if rate >= 24:
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R24_1, self.DataConsistency_R24_1)
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R24_2, self.DataConsistency_R24_2)
        if rate >= 20:
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R20_1, self.DataConsistency_R20_1)
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R20_2, self.DataConsistency_R20_2)
        if rate >= 16:
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R16_1, self.DataConsistency_R16_1)
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R16_2, self.DataConsistency_R16_2)
        if rate >= 12:
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R12_1, self.DataConsistency_R12_1)
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R12_2, self.DataConsistency_R12_2)
        if rate >= 8:
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R8_1, self.DataConsistency_R8_1)
            recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R8_2, self.DataConsistency_R8_2)

        # rate 4 for sure 
        recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R4_1, self.DataConsistency_R4_1)
        recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R4_2, self.DataConsistency_R4_2)
        recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R4_3, self.DataConsistency_R4_3)
        recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R4_4, self.DataConsistency_R4_4)
        recon = self.UnrolledBlock( recon, zerofilled, coil, mask, maskEmb, self.Network_R4_5, self.DataConsistency_R4_5)

        return recon


# testing, if correct it should give a network description
'''
if __name__ == '__main__':
    net = ZcUnrolledNet(1,0.001,5)
    print(net)
    # check parameters 
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
'''

if __name__ == '__main__':

    net = ZcUnrolledNet(0.001)

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    print('Network has ', get_n_params(net),' parameters')
