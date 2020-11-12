import torch
import torch.nn as nn
from torch.nn import init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None :
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None :
            nn.init.zeros_(m.bias)


def get_norm_layer(type_norm='none'):
    if type_norm == 'batch' :
        norm = nn.BatchNorm2d
        bias = False
    elif type_norm == 'instance' :
        norm = nn.InstanceNorm2d
        bias = True
    elif type_norm == 'none' :
        norm = nn.Identity
        bias = True
    else :
        raise NotImplementedError('%s: invalid normalization type'%(type_norm))
    return norm, bias


class ResidualBlock(nn.Module):
    def __init__(self, ch_inp, padding_mode='reflect', type_norm='none'):
        super(ResidualBlock, self).__init__()

        norm, bias = get_norm_layer(type_norm)

        block = [nn.Conv2d(ch_inp, ch_inp, kernel_size=3, stride=1,
                           padding=1, padding_mode='reflect', bias=bias),
                 norm(ch_inp), nn.ReLU(True),
                 nn.Conv2d(ch_inp, ch_inp, kernel_size=3, stride=1,
                           padding=1, padding_mode='reflect', bias=bias),
                 norm(ch_inp)]
        self.block = nn.Sequential(*block)

    def forward(self, inp):
        out = inp + self.block(inp)
        return out


class ResidualGenerator(nn.Module):
    def __init__(self, opt, ngf=64, nb_down=2, nb_block=9,
                 padding_mode='reflect', use_tanh=False):
        
        assert(nb_down>0 and nb_block>=0)
        super(ResidualGenerator, self).__init__()

        norm, bias = get_norm_layer(opt.type_norm)

        nb_feature = opt.ch_inp
        nb_feature_next = ngf
        block = []
        block += [nn.Conv2d(nb_feature, nb_feature_next, kernel_size=7,
                            stride=1, padding=3, padding_mode='reflect', bias=bias),
                  norm(nb_feature_next), nn.ReLU(True)]
        nb_feature = nb_feature_next

        for i in range(nb_down):
            nb_feature_next = nb_feature * 2
            block += [nn.Conv2d(nb_feature, nb_feature_next, kernel_size=3,
                                stride=2, padding=1, bias=bias),
                      norm(nb_feature_next), nn.ReLU(True)]
            nb_feature = nb_feature_next

        for j in range(nb_block):
            block += [ResidualBlock(nb_feature, padding_mode=padding_mode, type_norm=opt.type_norm)]

        for k in range(nb_down):
            nb_feature_next = nb_feature // 2
            block += [nn.ConvTranspose2d(nb_feature, nb_feature_next, kernel_size=3,
                                         stride=2, padding=1, output_padding=1, bias=bias),
                      norm(nb_feature_next), nn.ReLU(True)]
            nb_feature = nb_feature_next

        nb_feature_next = opt.ch_tar
        block += [nn.Conv2d(nb_feature, nb_feature_next, kernel_size=7,
                            stride=1, padding=3, padding_mode='reflect', bias=True)]

        if use_tanh :
            block += [nn.Tanh()]
            
        self.block = nn.Sequential(*block)
        print(self)

    def forward(self, inp):
        return self.block(inp)
    

class PixelDiscriminator(nn.Module):
    def __init__(self, opt, ndf=64, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()

        norm, bias = get_norm_layer(opt.type_norm)

        block = [nn.Conv2d(opt.ch_inp+opt.ch_tar, ndf, kernel_size=1, stride=1, padding=0, bias=True),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0, bias=bias),
                 norm(ndf*2), nn.LeakyReLU(0.2, True),
                 nn.Conv2d(ndf*2, 1, kernel_size=1, stride=1, padding=0, bias=True)]

        if use_sigmoid :
            block += [nn.Sigmoid()]
        self.block = nn.Sequential(*block)

    def forward(self, inp):
        return self.block(inp)


class PatchDiscriminator(nn.Module):
    def __init__(self, opt, nb_layer=3, ndf=64, ndf_max=512, use_sigmoid=False):
        super(PatchDiscriminator, self).__init__()

        norm, bias = get_norm_layer(opt.type_norm)

        blocks = []

        nb_feature = opt.ch_inp + opt.ch_tar
        nb_feature_next = ndf
        block = [nn.Conv2d(nb_feature, nb_feature_next, kernel_size=4,
                           stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2, True)]
        blocks.append(block)
        nb_feature = nb_feature_next

        for n in range(nb_layer - 1):
            nb_feature_next = min(nb_feature*2, ndf_max)
            block = [nn.Conv2d(nb_feature, nb_feature_next, kernel_size=4,
                               stride=2, padding=1, bias=bias),
                     norm(nb_feature_next), nn.LeakyReLU(0.2, True)]
            blocks.append(block)
            nb_feature = nb_feature_next

        nb_feature_next = min(nb_feature*2, ndf_max)
        block = [nn.Conv2d(nb_feature, nb_feature_next, kernel_size=4,
                           stride=1, padding=1, bias=bias),
                 norm(nb_feature_next), nn.LeakyReLU(0.2, True)]
        blocks.append(block)
        nb_feature = nb_feature_next

        block = [nn.Conv2d(nb_feature, 1, kernel_size=4,
                           stride=1, padding=1, bias=True)]
        if use_sigmoid :
            block += [nn.Sigmoid()]
        blocks.append(block)
        self.nb_blocks = len(blocks)
        for i in range(self.nb_blocks):
            setattr(self, 'block_%d'%(i), nn.Sequential(*blocks[i]))

    def forward(self, inp):
        result = [inp]
        for n in range(self.nb_blocks):
            block = getattr(self, 'block_%d'%(n))
            result.append(block(result[-1]))
        return result[1:]


class MultiPatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MultiPatchDiscriminator, self).__init__()
        self.nb_D = opt.nb_D
        for n in range(opt.nb_D):
            setattr(self, 'Discriminator_%d'%(n), PatchDiscriminator(opt))
        print(self)
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))
    def forward(self, inp):
        result = []
        for n in range(self.nb_D):
            if n != 0 :
                inp = nn.AvgPool2d(kernel_size=3, padding=1, stride=2) (inp)
            result.append(getattr(self, 'Discriminator_%d'%(n))(inp))
        return result


class Loss:
    def __init__(self, opt, device):

        self.device = device
        self.nb_D = opt.nb_D
        self.weight_FM_loss = opt.weight_FM_loss

        self.criterion_GAN = nn.MSELoss().to(self.device)
        self.criterion_FM = nn.L1Loss().to(self.device)

    def __call__(self, network_D, network_G, inp, tar):
        loss_D_real = 0
        loss_D_fake = 0
        loss_G_fake = 0
        loss_G_FM = 0
        loss_G = 0

        gen = network_G(inp)

        outputs_D_real = network_D(torch.cat([inp, tar], 1))
        outputs_D_fake = network_D(torch.cat([inp, gen.detach()], 1))
        for n in range(self.nb_D):
            output_D_real = outputs_D_real[n][-1]
            output_D_fake = outputs_D_fake[n][-1]
            target_D_real = torch.ones_like(output_D_real, dtype=torch.float).to(self.device)
            target_D_fake = torch.zeros_like(output_D_fake, dtype=torch.float).to(self.device)
            loss_D_real += self.criterion_GAN(output_D_real, target_D_real)
            loss_D_fake += self.criterion_GAN(output_D_fake, target_D_fake)
        loss_D = (loss_D_real + loss_D_fake)/2.

        outputs_G_fake = network_D(torch.cat([inp, gen], 1))
        for n in range(self.nb_D):
            output_G_fake = outputs_G_fake[n][-1]
            target_G_fake = torch.ones_like(output_G_fake, dtype=torch.float).to(self.device)
            loss_G_fake += self.criterion_GAN(output_G_fake, target_G_fake)

            features_real = outputs_D_real[n][:-1]
            features_fake = outputs_G_fake[n][:-1]

            for m in range(len(features_real)):
                loss_G_FM += self.criterion_FM(features_fake[m], features_real[m].detach())
        loss_G_FM *= (self.weight_FM_loss/self.nb_D)

        return gen, loss_D, loss_G_fake, loss_G_FM