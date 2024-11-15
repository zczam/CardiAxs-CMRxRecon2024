
import ZcUnrolledNet2D as UnrolledNetwork
import ZcDataLoader as FeedData

from torch import optim
import torch.nn as nn
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from time import perf_counter as timer
import ZcSSIM2D

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def ZcSSIM2D_for_3D(recon, Ref, SSIMfunc):
    # Data shape = [1, time, 1, ky, kx]
    recon = recon/torch.max(recon)
    Ref = Ref/torch.max(Ref)
    ssimVal = SSIMfunc(recon, Ref)
    return ssimVal

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable_num = ',trainable_num)
    print('Total var num = ',total_num)
    return 1


def ZcMixedLoss(recon, label):
    # this challenge measures the performance via PSNR, NMSE and SSIM.
    # since PSNR and NMSE are essentially the same thing, lets do NMSE and SSIM as loss.

    # NMSE
    recon = recon/recon.max()
    label = label/label.max()

    # SSIM

    return 
    #return torch.norm(recon-label,p=2)/torch.norm(label,p=2)


def Training(network, device, image_path, rate, epochs=200, batch_size=1, LearningRate=1e-4):

    CartesianData = FeedData.ZcDataLoader(datapath = image_path, rate=rate)
    Data_sampler = torch.utils.data.RandomSampler(CartesianData)
    data_loader = torch.utils.data.DataLoader(dataset=CartesianData,
                                               batch_size=batch_size, 
                                               #shuffle=True)  # here we use sampler to test distributed version
                                               sampler = Data_sampler,
                                               num_workers = 4)

    optimizer = optim.Adam(network.parameters(), lr=LearningRate)

    # best loss set to inf as starting point
    best_loss = float('inf')
    loss_List = []
    loss_List_Uniform = []
    loss_List_Grandom = []
    loss_List_Radial = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3600*3, gamma = 0.95)

    ssimLossFunc = ZcSSIM2D_for_3D
    ssimfunc = ZcSSIM2D.ZcSSIM2D()
    #plt.figure()
    # epoches
    iteration = 0
    for epoch in range(epochs):
        # start torch.nn.module's training mode
        network.train()
        loss_buff = []
        loss_buff_Uniform = []
        loss_buff_Grandom = []
        loss_buff_Radial = []

        # for loop over batches
        
        for zf, ref, coil, psf, mask, scale, maskEmb, mask_option, actual_rate, data_route in data_loader:
            
            # TIK!
            start_time = timer()

            iteration += 1
            # do zero_grad before every iteration
            optimizer.zero_grad()
            
            # get the data from data loader. 
            # notice the data loader won't really load anything before calling
            # image and label are sense1 image, real valued in size of [1, 2, 192,192,192]. 2 is real-imag channels
            
            zf = zf.cuda()
            ref = ref.cuda()
            coil = coil.cuda()
            mask = mask.cuda()
            psf = psf.cuda()
            maskEmb = maskEmb.cuda()
            # get recon
            recon = network(zf, coil, mask, maskEmb)

            # move to abs image to compute loss
            recon = torch.abs(recon[:,0,:,:,:] + recon[:,1,:,:,:]*1j)
            # get loss
            NMSEloss = torch.norm(recon-ref,p=2)/torch.norm(ref,p=2) 
            ys = int(recon.shape[-2]//4)
            ye = int(recon.shape[-2]//4*3)
            xs = int(recon.shape[-1]//3)
            xe = int(recon.shape[-1]//3*2)
            ssim_loss = ssimLossFunc(recon.permute(1,0,2,3)[:,:,ys:ye,xs:xe], ref[0,:,:,:,:].permute(1,0,2,3)[:,:,ys:ye,xs:xe], ssimfunc)

            loss = NMSEloss + (1-ssim_loss)

            #print('mean / max abs of ref = ', torch.mean(torch.abs(ref)).cpu().detach().numpy(), ' / ', torch.max(torch.abs(ref)).cpu().detach().numpy())
            #print('mean / max abs of zf  = ', torch.mean(torch.abs(zf)).cpu().detach().numpy(), ' / ', torch.max(torch.abs(zf)).cpu().detach().numpy())
            #print('mean / max abs of psf = ', torch.mean(torch.abs(psf)).cpu().detach().numpy(), ' / ', torch.max(torch.abs(psf)).cpu().detach().numpy())
            loss_buff = np.append(loss_buff, loss.item())
            if 'uniform' in mask_option[0]:
                loss_buff_Uniform = np.append(loss_buff_Uniform, loss.item())
            elif 'Grandom' in mask_option[0]:
                loss_buff_Grandom = np.append(loss_buff_Grandom, loss.item())
            else: 
                loss_buff_Radial = np.append(loss_buff_Radial, loss.item())

            # backpropagate
            loss.backward()
        
            # update parameters
            optimizer.step()
            scheduler.step()

            if iteration % 3600 == 0 and not torch.isnan(loss):
                checkpoint = { 
                    'epoch': epoch,
                    'model': network.state_dict(),
                    'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, 'ModelTemp/Model_Epoch%d_Iter%d.pth'%(epoch,iteration))   

                sio.savemat('ModelTemp/Loss_Epoch%d.mat'%epoch,{'loss':np.mean(loss_buff),'lossU': np.mean(loss_buff_Uniform), 'lossG': np.mean(loss_buff_Grandom), 'lossR': np.mean(loss_buff_Radial)})  
            '''
            if 'P166/cine_sax_slice8_ksp.mat' in data_route[0]:
                sio.savemat('ModelTemp/ReconP166CineSaxSlice8_'+mask_option[0]+'_R'+actual_rate[0]+'_Epoch%d.mat'%epoch, {'recon': recon.cpu().detach().numpy()})
                if epoch == 0: 
                    sio.savemat('ModelTemp/RefP166CineSaxSlice8Epoch%d.mat'%epoch, {'ref': ref.cpu().detach().numpy()})
            '''
            # Tok!
            end_time = timer()
            print('NMSE = %.3f'%NMSEloss.cpu().detach().numpy(),' / SSIM = %.3f'%ssim_loss.cpu().detach().numpy(),' / loss = %.3f'%loss.cpu().detach().numpy(), ', Sampling pattern == ',mask_option[0], ' rate ', actual_rate[0] ,', at iter ', iteration, '/', len(data_loader), ', at epoch #', epoch+1, '/', epochs, ' -- Time Cost = %.3f'%(end_time - start_time), ' sec')
        
        checkpoint = { 
            'epoch': epoch,
            'model': network.state_dict(),
            'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'ModelTemp/Model_Epoch%d_Iter%d.pth'%(epoch,iteration)) 

        loss_List = np.append(loss_List, np.mean(loss_buff)/2)    
        loss_List_Uniform = np.append(loss_List_Uniform, np.mean(loss_buff_Uniform)/2)
        loss_List_Grandom = np.append(loss_List_Grandom, np.mean(loss_buff_Grandom)/2)
        loss_List_Radial = np.append(loss_List_Radial, np.mean(loss_buff_Radial)/2)
        sio.savemat('LossCurve.mat',{'loss':loss_List,'lossU': loss_List_Uniform, 'lossG': loss_List_Grandom, 'lossR': loss_List_Radial})
        
            
    return 1
 

if __name__ == "__main__":
    
    rate = 4
    
    # check CUDA availiability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the network
    network = UnrolledNetwork.ZcUnrolledNet(0.05)
    #network.load_state_dict(torch.load('ModelTemp/Model_Epoch0_Iter162000.pth')['model'])
    # just have a look what we have 
    network = network.cuda()
    for i in network.state_dict():
        print(i)
    
    imageroute = '/home/chi/MICCAI2024Challenge/home2/ForTraining/MICCAIChallenge2024/ChallengeData/MultiCoil/'

    Training(network, device, imageroute, rate)
