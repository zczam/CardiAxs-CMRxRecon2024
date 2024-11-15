
import scipy.io as sio
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import h5py
import sys

bart_path = "bart/"
os.environ["TOOLBOX_PATH"] = bart_path
sys.path.append(os.path.join(bart_path, 'python'))
import bart

# kooshball loader.
# to save the storage, kooshball data are stored as 1 channel SENSE1, complex64
# coil will be stored individually. 
# Thus multi-coil images are obtained as SENSE1 * coil.
# fast operator is also individually saved. 
# for convinience testing different rates, input images and labels are stored individually
# undersampled images are stored under root/r*./Subject*Data*.mat, variable is called zerofilled.
# labels are root/label/Subject*label*.mat, variable is called label.
# coils are stored for each subject, under root/coil/Subject*.mat

class ZcDataLoader(Dataset):
    
    def __init__(self, datapath):

        # kspace route e.g.
        # /home/chi/MICCAI2024Challenge/  home2/ForTraining  /MICCAIChallenge2024/ChallengeData/MultiCoil/  Cine/  TrainingSet/FullSample  /P001/  *.mat
        self.h5path = glob.glob(os.path.join(datapath, '*/TestSet/UnderSample_Task2/*/*.mat'))
        
        #self.ksp_path_all = glob.glob(os.path.join(datapath, 'Aorta/ValidationSet/UnderSample_Task2/P003/*_ksp.mat'))
        #self.ksp_path_all = glob.glob(os.path.join(datapath, 'Mapping/ValidationSet/UnderSample_Task2/P001/*_ksp.mat'))

    def __GetCoilMaskEtc__(self, h5path):

        file = h5py.File(h5path, 'r')['kus']
        #file = sio.loadmat(h5path)
        ksp = file['real'] + file['imag'] * 1j

        [time, sli, nc, ky, kx] = ksp.shape

        # for single-slice, data shapes are:
        # kspace.shape = [Time, Coil, Ky, Kx]
        # coil.shape = [1, Coil, Ky, Kx]
        # for multi-slice, make them like:
        # kspace_all.shape = [slice, Time, Coil, ky, kx]ss
        # coil_all.shape = [slice, 1, Coil, ky, kx]
        kspace_all = np.zeros(shape = [sli, time, nc, ky, kx], dtype = ksp.dtype)
        coil_all = np.zeros(shape = [sli, 1, nc, ky, kx], dtype = ksp.dtype)
        for sl in range(sli):
            kspace = ksp[:,sl,:,:,:]
            
            eps = np.finfo(kspace.dtype).eps
            kspace_avg = np.sum(kspace, axis=0) / (np.sum(kspace != 0, axis=0) + eps)
            kspace_avg = np.expand_dims(kspace_avg.transpose(2,1,0),-2)
            print('ESPIRiT On Going! %d/%d'%(sl+1,sli))
            coil = bart.bart(1, f'ecalib -S -m {1} -r {16} -c {0.1}', kspace_avg)
            #coil = bart.bart(1, 'ecalib -S -m 1 -r 16 -c 0.1', kspace_avg)
            coil = np.squeeze(coil).transpose(2,1,0)

            kspace_all[sl, :,:,:,:] = np.copy(kspace)
            coil_all[sl, 0,:,:,:] = np.copy(coil)

        return kspace_all, coil_all

    def __getitem__(self, index):
        
        # 0. Arrange index
        # each image has unknown number of slices, with 3 kinds of sampling patterns for each 2D k-t image: uniform, random and radial. 
        # thus we have N 2D k-t single slice images, we need __len__(self) to be N*3 to iterate all possible patterns 
        # each pattern has 6 rates, in total N*3*6 masks
        # We select the mask by index, index % 3 == 0 --> uniform, == 1 --> random, == 2 --> radial. in total index = 0:3N
        
        # 1. Get kspace and coil 
        kspace_all, coil_all = self.__GetCoilMaskEtc__(self.h5path[index])
        # coil.shape = [Coil, Ky, Kx] --> Averaged from Time axis. After padding shape == [1, Coil, ky, kx]

        # 2. Get mask, firstly determine the image type & view
        # e.g. /*/cine_lax_slice0_ksp.mat --> type == cine, view == lax
        data_type = self.h5path[index].split('/')[-1].split('_')[0] + '_'
        data_view = self.h5path[index].split('/')[-1].split('_')[1] + '_'
        mask_type = self.h5path[index].split('/')[-1].split('_kus_')[1].split('_')[0]
        selected_rate = [int(aa) for aa in mask_type if aa.isdigit()]
        selected_rate = int(''.join(str(x) for x in selected_rate))
        # some images have no view options 
        if 'kus' in data_view:
            data_view = ''

        if 'ktUniform' in mask_type:
            mask_option = 'uniform'
        elif 'ktGaussian' in mask_type:
            mask_option = 'Grandom'
        else: 
            mask_option = 'radial'
        mask_name = data_type + data_view + 'mask_' + mask_type
        
        # then we move to the mask route. 
        # mask_route   = /home/chi/MICCAI2024Challenge/  home2/ForTraining/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/  Mask_Task2  /P001/  cine_lax_mask_Uniform4.mat
        # kspace_route = /home/chi/MICCAI2024Challenge/  home2/ForTraining/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/  FullSample  /P001/  cine_lax_slice0_ksp.mat        
        mask_route = self.h5path[index].replace('UnderSample_Task2', 'Mask_Task2')
        mask_route = os.path.dirname(mask_route) + '/' + mask_name

        #mask_route = mask_route.replace('TestSet','ValidationSet')

        MaskFile = h5py.File(mask_route, 'r')
        mask = np.expand_dims(np.float32(np.array(MaskFile['mask'])),1)
        
        #mask = np.float32(np.expand_dims(sio.loadmat(mask_route)['mask'], 1))
        #kspace_all = kspace_all*mask
        
        # mask.shape = [time, ky, kx], after padding shape == [time, 1, ky, kx]


        # 3. Get the materials we need. 
        # Reference: label image, IFFT of kspace + coil comb
        # Zero-filled: input image, IFFT of kspace .* mask + coil comb
        # PSF: IFFT of mask
        # mask, coil etc: just return as they are. 

        # zero-filled
        # kspace_all.shape = [slice, Time, Coil, ky, kx]
        # coil_all.shape = [slice, 1, Coil, ky, kx]
        [nsli, ntime, ncoil, ky, kx] = kspace_all.shape
        zf_all = np.zeros(shape = [nsli, ntime, 2, ky, kx], dtype = np.float32)
        img_scale_all = np.zeros(shape = [nsli], dtype = np.float32)
        for sliceIdx in range(kspace_all.shape[0]):
            zf = np.complex64(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_all[sliceIdx,:,:,:,:],axes = [-1, -2]),axes = [-1, -2], norm = 'ortho'),axes = [-1, -2]))
            zf = np.sum(zf*np.conjugate(coil_all[sliceIdx,:,:,:,:]),axis = -3, keepdims = True)
            zf = np.concatenate([np.real(zf), np.imag(zf)], axis=-3)
            img_scale = np.max(np.abs(zf))
            zf = zf/img_scale
            zf_all[sliceIdx,:,:,:,:] = zf
            img_scale_all[sliceIdx] = img_scale
       

        maskEmb = self.__GetMaskEmb__(mask)
        # This version needs everything to be in [slice, channel (coil), time, y, x]
        zf_all = zf_all.transpose(0,2,1,3,4)
        coil_all = coil_all.transpose(0,2,1,3,4)
        mask = mask.transpose(1,0,2,3)        
        # all data in the shape of [time, coil, ky, kx]. We treat it as Time x multi-coil 2D, for the convience in DC part since coil is shared
        # ref, zf and psf has [time, 2, ky, kx], where coil dim is real-imag channel. 
        
        return zf_all, coil_all, mask, img_scale_all, maskEmb, str(selected_rate), self.h5path[index]
         

    def __GetMaskEmb__(self, mask):

        [Batch, Ch, Y, X] = mask.shape
        mask_left = mask[:,:,:mask.shape[-2]//2,:]
        mask_right = mask[:,:,mask.shape[-2]//2:,:]

        rate_L_M = np.zeros([Batch,1], dtype = np.float32)
        rate_R_M = np.zeros([Batch,1], dtype = np.float32)
        rate_L_S = np.zeros([Batch,1], dtype = np.float32)
        rate_R_S = np.zeros([Batch,1], dtype = np.float32)

        rate_L_M2 = np.zeros([Batch,1], dtype = np.float32)
        rate_R_M2 = np.zeros([Batch,1], dtype = np.float32)
        rate_L_S2 = np.zeros([Batch,1], dtype = np.float32)
        rate_R_S2 = np.zeros([Batch,1], dtype = np.float32)

        for Idx in range(Batch):

            mask_left_sum1 = mask_left[Idx, :].sum(-1)
            mask_left_sum2 = mask_left[Idx, :].sum(-2)
            mask_right_sum1 = mask_right[Idx, :].sum(-1)
            mask_right_sum2 = mask_right[Idx, :].sum(-2)

            sampledPosSpaceL = np.diff(mask_left_sum1.nonzero()[-1].astype(np.float32))
            sampledPosSpaceR = np.diff(mask_right_sum1.nonzero()[-1].astype(np.float32))

            sampledPosSpaceL2 = (mask_left_sum2.astype(np.float32))
            sampledPosSpaceR2 = (mask_right_sum2.astype(np.float32))

            rate_L_M[Idx,0] = sampledPosSpaceL.mean()
            rate_R_M[Idx,0] = sampledPosSpaceR.mean()
            rate_L_S[Idx,0] = sampledPosSpaceL.std()
            rate_R_S[Idx,0] = sampledPosSpaceR.std()

            rate_L_M2[Idx,0] = sampledPosSpaceL2.mean()
            rate_R_M2[Idx,0] = sampledPosSpaceR2.mean()
            rate_L_S2[Idx,0] = sampledPosSpaceL2.std()
            rate_R_S2[Idx,0] = sampledPosSpaceR2.std()
        
        return np.concatenate((rate_L_M,rate_R_M,rate_L_S,rate_R_S, rate_L_M2,rate_R_M2,rate_L_S2,rate_R_S2), -1) 



    def __len__(self):
        return len(self.h5path)

    
