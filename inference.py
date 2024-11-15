
import ZcUnrolledNet2D as UnrolledNetwork
import ZcDataLoader_Val_NoFS as FeedData

import torch
import numpy as np
import scipy.io as sio
import os

import sys
import argparse

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def crop_data_ML(im, datatype):

    if datatype == 'BlackBlood':
        [sz, sy, sx] = im.shape
        st = 1
    else:
        [st, sz, sy, sx] = im.shape

    if sz < 3:
        sz0 = 0
        sz1 = int(sz)
    else:
        sz0 = int(np.round(sz/2+1e-8)-2)
        sz1 = int(np.round(sz/2+1e-8))
        
        
    if datatype == 'BlackBlood':
        st0 = 0
        st1 = 1
    elif datatype == 'Mapping':
        st0 = 0
        st1 = int(st)
    else:
        st0 = 0
        st1 = 3
        
    sy0 = int(np.floor(sy/2)) + int(np.ceil(-sy/4))
    sy1 = int(np.floor(sy/2)) + int(np.ceil(sy/4))

    sx0 = int(np.floor(sx/2)) + int(np.ceil(-sx/6))
    sx1 = int(np.floor(sx/2)) + int(np.ceil(sx/6))
    
    if datatype == 'BlackBlood':
        imc = im[sz0:sz1, sy0:sy1, sx0:sx1]
    else:
        imc = im[st0:st1, sz0:sz1, sy0:sy1, sx0:sx1]
        
    return imc



def Validation(network, input_dir, output_dir):

    CartesianData = FeedData.ZcDataLoader(datapath = input_dir)
    Data_sampler = torch.utils.data.RandomSampler(CartesianData)
    data_loader = torch.utils.data.DataLoader(dataset=CartesianData,
                                               batch_size=1, 
                                               #shuffle=True)  # here we use sampler to test distributed version
                                               sampler = Data_sampler,
                                               num_workers = 1)


    #plt.figure()
    # epoches
    # start torch.nn.module's training mode
    network.eval()

    # for loop over batches
    for zf_all, coil_all, mask, scale_all, maskEmb, actual_rate, data_route in data_loader:
        
        mask = mask.cuda()
        recon_all = torch.zeros_like(zf_all)
        maskEmb = maskEmb.cuda()
        # get the data from data loader. 
        # notice the data loader won't really load anything before calling
        # image and label are sense1 image, real valued in size of [1, 2, 192,192,192]. 2 is real-imag channels
        for sliceIdx in range(zf_all.shape[1]):
            zf = zf_all[:,sliceIdx,:,:,:,:].cuda()
            coil = coil_all[:,sliceIdx,:,:,:,:].cuda()

            scale = scale_all[0,sliceIdx].cuda()
            # get recon
            recon_all[:,sliceIdx,:,:,:,:] = network(zf, coil, mask, maskEmb, int(actual_rate[0])) * scale
        
        # move to abs image to compute loss
        recon_all = torch.abs(recon_all[0,:,0,:,:,:] + recon_all[0,:,1,:,:,:]*1j).cpu().detach().numpy()
        # recon_all is saved as |coil-combined image|, in shape of [slice, time, ky, kx]
        # shape needs to be  [sx,sy,scc,sz,t] = size(img); --> x, y, coil, slice, time before SOS
        # after SOS it has no scc, --> [sx, sy, sz, t]
        #print('recon_all.shape = ', recon_all.shape)
        recon_all = recon_all.transpose([1,0,2,3])
        #print('after permute recon_all.shape = ', recon_all.shape)
        data_route = data_route[0]
        Modality = data_route.split('/')[-5]
        img4ranking = recon_all.transpose(3,2,1,0)
        #img4ranking = crop_data_ML(recon_all, Modality).transpose(3,2,1,0)
        SaveName = data_route.replace(input_dir,output_dir + '/').replace('UnderSample_Task2','Task2')
        SaveFolder = os.path.dirname(os.path.abspath(SaveName))
        if not os.path.exists(SaveFolder):
            os.makedirs(SaveFolder)
        
        print('SaveName = ', SaveName, ', shape = ', img4ranking.shape, ' after cropping')
        sio.savemat(SaveName, {'img4ranking':img4ranking})

    return 1




def Go(input_dir, output_dir):

    Epoch = 1
    # check CUDA availiability

    # initialize the network
    network = UnrolledNetwork.ZcUnrolledNet(0.05)
    network.load_state_dict(torch.load('TheModel.pth')['model'])
    # just have a look what we have 
    network = network.cuda()
    
    Validation(network, input_dir, output_dir)
    
    #FormatProcess4Submit(SaveRoute)

if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    Go(input_dir, output_dir)

    print('Done. GLHF')