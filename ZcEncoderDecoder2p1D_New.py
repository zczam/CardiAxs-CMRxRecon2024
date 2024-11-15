import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv2DThen1D(nn.Module):
    def __init__(self, in_channels, out_channels_2d, out_channels_1d, kernel_size = 3, stride= 1, bias = False):
        super(Conv2DThen1D, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels_2d, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='replicate', bias = bias, stride = stride) # we cannot support full time dim here. in case time frames might end suddenly, we might not use circular padding 
        self.conv1d = nn.Conv1d(out_channels_2d, out_channels_1d, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='replicate', bias = bias)
        

    def forward(self, x):
        batch_size, channels, dz, dy, dx = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [batch, z, channels, y, x]
        x = x.reshape(batch_size * dz, channels, dy, dx) # merge z & batch into batch
        x = self.conv2d(x)  # Apply 2D conv
        [batchz, ch, dy, dx] = x.shape
        x = x.reshape(batch_size, dz, ch, dy, dx) # split z & batch
        x = x.permute(0, 3, 4, 2, 1)  # [batch, y, x, channels, z]
        x = x.reshape(batch_size*dy*dx, ch, dz) # merge to [batch, chan, z]
        x = self.conv1d(x)  # Apply 1D conv
        [bxy, ch, dz] = x.shape
        x = x.reshape(batch_size,dy,dx, ch, dz) # split to [batch, y,x,chan,z]
        x = x.permute(0, 3, 4, 1, 2)  # [batch, channels_1d, z, y, x]
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            #nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            Conv2DThen1D(channel, channel // reduction, channel // reduction, kernel_size = 1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            Conv2DThen1D(channel // reduction, channel // reduction, channel, kernel_size = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super(CAB, self).__init__()
        modules_body = []
        #modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(Conv2DThen1D(n_feat, n_feat, n_feat, kernel_size = kernel_size))
        modules_body.append(act)
        modules_body.append(Conv2DThen1D(n_feat, n_feat, n_feat, kernel_size = kernel_size))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_input_prompt = False):
        super(PromptBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        #self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec_conv3x3 = Conv2DThen1D(prompt_dim,prompt_dim,prompt_dim)

    def forward(self, x):
        [batch, Chan, dz, dy, dx] = x.shape
        x = x.permute(0,2,1,3,4).reshape(batch*dz, Chan, dy, dx)
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        [bz, Chan, dy, dx] = prompt.shape
        prompt = prompt.reshape([batch, dz, Chan, dy, dx]).permute([0,2,1,3,4])
        prompt = self.dec_conv3x3(prompt)

        return prompt

class PatternPromptBlock(nn.Module):
    def __init__(self, featuremap_dim = 64, prompt_dim=128, prompt_len=18, prompt_size=96, lin_dim=192, learnable_input_prompt = False):
        super(PatternPromptBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layerUp = nn.Linear(8, lin_dim)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        #self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec_conv3x3 = Conv2DThen1D(prompt_dim+featuremap_dim,prompt_dim,prompt_dim)

    def forward(self, x, emb):
        [batch, Chan, dz, H, W] = x.shape
        B = dz*batch
        emb = emb[0,:,:] # input emb shape = 1, batch (Time), 4
        prompt_weights = F.softmax(self.linear_layer(self.linear_layerUp(emb)), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        [bz, Chan, dy, dx] = prompt.shape
        prompt = prompt.reshape([batch, dz, Chan, dy, dx]).permute([0,2,1,3,4])
        prompt = self.dec_conv3x3(torch.cat((x,prompt), 1))

        return prompt


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act, no_use_ca=False, first_act=False):
        super(DownBlock, self).__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction, bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(*(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        #self.down = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.down = Conv2DThen1D(input_channel, output_channel, output_channel, stride=2, bias = True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc

class UpSample2Dfor5Ddata(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(UpSample2Dfor5Ddata, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        self.scale_factor = scale_factor
    def forward(self, x):
        [batch, chan, dz, dy, dx] = x.shape
        x = x.permute([0,2,1,3,4]).reshape([batch*dz, chan, dy, dx])
        x = self.upsample(x)
        [bz, chan, dy2, dx2] = x.shape
        return x.reshape([batch, dz, chan, dy2, dx2]).permute([0,2,1,3,4])
        


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, promptP_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super(UpBlock, self).__init__()
        prompt_dim2 = prompt_dim + promptP_dim
        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim2, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        #self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)
        self.reduce = Conv2DThen1D(in_dim+prompt_dim2, in_dim+prompt_dim2, in_dim, kernel_size=1)

        self.up = nn.Sequential(UpSample2Dfor5Ddata(scale_factor=2, mode='bilinear', align_corners=False),
                                Conv2DThen1D(in_dim, in_dim, out_dim, kernel_size=1) )
                                #nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)


    def forward(self,x,prompt_dec,skip, prompt_pattern):
        #print('x.shape = ', x.shape)
        #print('prompt.shape = ', prompt_dec.shape)
        #print('prompt_pattern.shape = ', prompt_pattern.shape)
        x = torch.cat([x, prompt_dec, prompt_pattern], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)
        x = self.up(x)[:,:,:,:skip.shape[-2], :skip.shape[-1]] + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super(SkipBlock, self).__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(
                *[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)
        return x

class PromptUnet(nn.Module):
    def __init__(self, 
                 in_chans=2*3*3, 
                 out_chans=2, 
                 n_feat0=48,
                 feature_dim = [48, 72, 96],
                 prompt_dim = [24, 48, 72],
                 len_prompt = [5,5,5],
                 prompt_size = [64, 32, 16],
                 prompt_dimP = [36, 54, 72],
                 len_promptP = [18,18,18],
                 prompt_sizeP = [72, 54, 36],
                 n_enc_cab = [2, 3, 3],
                 n_dec_cab = [2, 2, 3],
                 n_skip_cab = [1, 1, 1],
                 n_bottleneck_cab = 3,
                 no_use_ca = False,
                 learnable_input_prompt=True,
                 kernel_size=3, 
                 reduction=4, 
                 act=nn.PReLU(), 
                 bias=False,
                 ):
        """
        PromptUnet, see in paper: https://arxiv.org/abs/2309.13839
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            n_feat0: Number of output channels in the first convolution layer.
            feature_dim: Number of output channels in each level of the encoder.
            prompt_dim: Number of channels in the prompt at each level of the decoder.
            len_prompt: number of components in the prompt at each level of the decoder.
            prompt_size: spatial size of the prompt at each level of the decoder.
            n_enc_cab: number of channel attention blocks (CAB) in each level of the encoder.
            n_dec_cab: number of channel attention blocks (CAB) in each level of the decoder.
            n_skip_cab: number of channel attention blocks (CAB) in each skip connection.
            n_bottleneck_cab: number of channel attention blocks (CAB) in the bottleneck.
            kernel_size: kernel size of the convolution layers.
            reduction: reduction factor for the channel attention blocks (CAB).
            act: activation function.
            bias: whether to use bias in the convolution layers.
            no_use_ca: whether to *not* use channel attention blocks (CAB).
            learnable_input_prompt: whether to learn the input prompt in the PromptBlock.
        """
        super(PromptUnet, self).__init__()

        # Feature extraction
        #self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)
        self.feat_extract = Conv2DThen1D(in_chans, n_feat0, n_feat0)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1p = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn2p = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn3p = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_bottleneck_cab)])

        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim=prompt_dim[2], prompt_len=len_prompt[2], prompt_size=prompt_size[2], lin_dim=feature_dim[2], learnable_input_prompt=learnable_input_prompt)
        self.prompt_level3p = PatternPromptBlock(featuremap_dim = feature_dim[2], prompt_dim=prompt_dimP[2], prompt_len=len_promptP[2], prompt_size=prompt_sizeP[2], lin_dim=feature_dim[2], learnable_input_prompt=learnable_input_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], prompt_dimP[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        self.prompt_level2 = PromptBlock(prompt_dim=prompt_dim[1], prompt_len=len_prompt[1], prompt_size=prompt_size[1], lin_dim=feature_dim[1], learnable_input_prompt=learnable_input_prompt)
        self.prompt_level2p = PatternPromptBlock(featuremap_dim = feature_dim[1], prompt_dim=prompt_dimP[1], prompt_len=len_promptP[1], prompt_size=prompt_sizeP[1], lin_dim=feature_dim[1], learnable_input_prompt=learnable_input_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], prompt_dimP[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        self.prompt_level1 = PromptBlock(prompt_dim=prompt_dim[0], prompt_len=len_prompt[0], prompt_size=prompt_size[0], lin_dim=feature_dim[0], learnable_input_prompt=learnable_input_prompt)
        self.prompt_level1p = PatternPromptBlock(featuremap_dim = feature_dim[0], prompt_dim=prompt_dimP[0], prompt_len=len_promptP[0], prompt_size=prompt_sizeP[0], lin_dim=feature_dim[0], learnable_input_prompt=learnable_input_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], prompt_dimP[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # OutConv
        #self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)
        self.conv_last = Conv2DThen1D(n_feat0, n_feat0, out_chans, kernel_size = 5)

    def forward(self, x, MaskEmb):

        x = torch.cat((x, \
                torch.roll(x,x.shape[-2]//3 * 1, -2), \
                torch.roll(x,x.shape[-2]//3 * 2, -2), \
                ), 1)  # ky channel shift --> ch = 4*3 = 12
        x = torch.cat((x, \
                       torch.roll(x,x.shape[-1]//3 * 1, -1), \
                       torch.roll(x,x.shape[-1]//3 * 2, -1), \
                       ), 1) # kx channel-shift --> ch = 12*3 = 36
        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        dec_prompt3 = self.prompt_level3(x)
        dec_prompt3p = self.prompt_level3p(x, MaskEmb)
        x = self.dec_level3(x,dec_prompt3,self.skip_attn3(enc3), dec_prompt3p)

        dec_prompt2 = self.prompt_level2(x)
        dec_prompt2p = self.prompt_level2p(x, MaskEmb)
        x = self.dec_level2(x,dec_prompt2,self.skip_attn2(enc2), dec_prompt2p)

        dec_prompt1 = self.prompt_level1(x)
        dec_prompt1p = self.prompt_level1p(x, MaskEmb)
        x = self.dec_level1(x,dec_prompt1,self.skip_attn1(enc1), dec_prompt1p)

        # 4. last conv
        return self.conv_last(x)
    



if __name__ == '__main__':

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Trainable_num = ',trainable_num)
        print('Total var num = ',total_num)
        return total_num, trainable_num

    net2 = PromptUnet(in_chans = 2, out_chans = 2)
    #print(net)


    
    # check parameters 
    #for name, param in net.named_parameters():
     #   print(name, param.size(), type(param))

    input = torch.zeros([1,2,3,256,256], dtype = torch.float32)
    emb = torch.ones([1,3,8], dtype = torch.float32)
    output2 = net2(input,emb)

    get_parameter_number(net2)

