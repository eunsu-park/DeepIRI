import torch
import torch.nn as nn
from torch.nn import init
from functools import partial

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
        norm = partial(nn.BatchNorm2d, momentum=0.9, affine=True, eps=1.01e-5)
    elif type_norm == 'instance' :
        norm = partial(nn.InstanceNorm2d, momentum=0.9, affine=False, eps=1.01e-5)
    elif type_norm == 'none' :
        norm = nn.Identity
    else :
        raise NotImplementedError('%s: invalid normalization type'%(type_norm))
    return norm


class ResidualBlock(nn.Module):
    def __init__(self, nb_feat, norm):
        super(ResidualBlock, self).__init__()

        self.build(nb_feat, norm)

    def build(self, nb_feat, norm):
        block = [nn.Conv2d(nb_feat, nb_feat, kernel_size=3, stride=1,
                           padding=1, padding_mode='reflect'),
                 norm(nb_feat), nn.ReLU(),
                 nn.Conv2d(nb_feat, nb_feat, kernel_size=3, stride=1,
                           padding=1, padding_mode='reflect'),
                 norm(nb_feat)]
        self.block = nn.Sequential(*block)

    def forward(self, inp):
        return inp + self.block(inp)


class ResidualGenerator(nn.Module):
    def __init__(self, opt):
        super(ResidualGenerator, self).__init__()

        self.opt = opt
        self.build()
        print(self)

    def build(self):
        ch_inp = self.opt.ch_inp
        ch_tar = self.opt.ch_tar
        nb_feat = self.opt.nb_feat_init_G
        nb_down = self.opt.nb_down
        nb_block = self.opt.nb_block
        use_tanh = self.opt.use_tanh
        type_norm = self.opt.type_norm
        norm = get_norm_layer(type_norm)

        block = []
        block += [nn.Conv2d(ch_inp, nb_feat, kernel_size=7,
                            stride=1, padding=3, padding_mode='reflect'),
                  norm(nb_feat), nn.ReLU()]

        for i in range(nb_down):
            block += [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=3,
                                stride=2, padding=1),
                      norm(nb_feat*2), nn.ReLU()]
            nb_feat *= 2

        for j in range(nb_block):
            block += [ResidualBlock(nb_feat, norm)]

        for k in range(nb_down):
            block += [nn.ConvTranspose2d(nb_feat, nb_feat//2, kernel_size=3,
                                         stride=2, padding=1, output_padding=1),
                      norm(nb_feat//2), nn.ReLU()]
            nb_feat //=2

        block += [nn.Conv2d(nb_feat, ch_tar, kernel_size=7,
                            stride=1, padding=3, padding_mode='reflect')]

        if use_tanh :
            block += [nn.Tanh()]
            
        self.block = nn.Sequential(*block)
        print(self)

    def forward(self, inp):
        return self.block(inp)
    

class PixelDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PixelDiscriminator, self).__init__()
        self.opt = opt
        self.build()
        print(self)

    def build(self):
        ch_inp = self.opt.ch_inp
        ch_tar = self.opt.ch_tar
        nb_feat = self.opt.nb_feat_init_D
        use_sigmoid = self.opt.use_sigmoid
        type_norm = self.opt.type_norm
        norm = get_norm_layer(type_norm)
        block = [nn.Conv2d(ch_inp+ch_tar, nb_feat, kernel_size=1, stride=1, padding=0),
                 nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat, nb_feat*2, kernel_size=1, stride=1, padding=0),
                 norm(nb_feat*2), nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat*2, 1, kernel_size=1, stride=1, padding=0)]
        if self.opt.use_sigmoid :
            block += [nn.Sigmoid()]
        self.block = nn.Sequential(*block)

    def forward(self, inp):
        return self.block(inp)


class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.opt = opt
        self.build()
        print(self)

    def build(self):
        ch_inp = self.opt.ch_inp
        ch_tar = self.opt.ch_tar
        nb_layer = self.opt.nb_layer
        use_sigmoid = self.opt.use_sigmoid
        nb_feat = self.opt.nb_feat_init_D
        type_norm = self.opt.type_norm
        norm = get_norm_layer(type_norm)
        blocks = []
        block = [nn.Conv2d(ch_inp+ch_tar, nb_feat, kernel_size=4,
                           stride=2, padding=1),
                 nn.LeakyReLU(0.2)]
        blocks.append(block)
        for n in range(nb_layer - 1):
            block = [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=4,
                               stride=2, padding=1),
                     norm(nb_feat*2), nn.LeakyReLU(0.2)]
            blocks.append(block)
            nb_feat *= 2
        block = [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=4,
                           stride=1, padding=1),
                 norm(nb_feat*2), nn.LeakyReLU(0.2)]
        blocks.append(block)
        nb_feat *= 2
        block = [nn.Conv2d(nb_feat, 1, kernel_size=4,
                           stride=1, padding=1)]
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

        if opt.type_gan == 'gan' :
            self.criterion = nn.BCELoss().to(self.device)
        elif opt.type_gan == 'lsgan' :
            self.criterion = nn.MSELoss().to(self.device)

        self.FMcriterion = nn.L1Loss().to(self.device)

    def __call__(self, network_D, network_G, inp, tar):
        loss_D = 0
        loss_G_fake = 0
        loss_G_FM = 0

        gen = network_G(inp)

        outputs_D_real = network_D(torch.cat([inp, tar], 1))
        outputs_D_fake = network_D(torch.cat([inp, gen.detach()], 1))

        for n in range(self.nb_D):
            output_D_real = outputs_D_real[n][-1]
            output_D_fake = outputs_D_fake[n][-1]
            target_D_real = torch.ones_like(output_D_real, dtype=torch.float).to(self.device)
            target_D_fake = torch.zeros_like(output_D_fake, dtype=torch.float).to(self.device)
            loss_D_real = self.criterion(output_D_real, target_D_real)
            loss_D_fake = self.criterion(output_D_fake, target_D_fake)
            loss_D += (loss_D_real+loss_D_fake)/2.

        outputs_G_fake = network_D(torch.cat([inp, gen], 1))
        for n in range(self.nb_D):
            output_G_fake = outputs_G_fake[n][-1]
            target_G_fake = torch.ones_like(output_G_fake, dtype=torch.float).to(self.device)
            loss_G_fake += self.criterion(output_G_fake, target_G_fake)

            features_real = outputs_D_real[n][:-1]
            features_fake = outputs_G_fake[n][:-1]
            for m in range(len(features_real)):
                loss_G_FM += self.FMcriterion(features_fake[m], features_real[m].detach())

        loss_G_FM *= (self.weight_FM_loss/self.nb_D)

        return gen, loss_D, loss_G_fake, loss_G_FM


if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()
#    network_D = PixelDiscriminator(opt)
#    network_D = PatchDiscriminator(opt)
    network_D = MultiPatchDiscriminator(opt)
    network_G = ResidualGenerator(opt)

    inp = torch.ones((1, 1, 128, 128))
    tar = torch.ones((1, 1, 128, 128))

    print(inp.shape, tar.shape)

    device = torch.device('cpu')
    loss = Loss(opt, device)
    gen, loss_D, loss_G_fake, loss_G_FM = loss(network_D, network_G, inp, tar)
    print(loss_D, loss_G_fake, loss_G_FM)



