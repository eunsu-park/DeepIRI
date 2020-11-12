from option import TrainOption, TestOption
opt = TrainOption().parse()

import torch
import torch.nn as nn
import random
import numpy as np

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchmark = False

import os, time, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
ngpu = torch.cuda.device_count()
cuda = ngpu > 0
device = torch.device('cuda' if cuda else 'cpu')
print(ngpu, device)
warnings.filterwarnings('ignore')

path_model = '%s/%s/model'%(opt.root_save, opt.prefix)
if not os.path.exists(path_model):
    os.makedirs(path_model)
path_snap = '%s/%s/snap'%(opt.root_save, opt.prefix)
if not os.path.exists(path_snap):
    os.makedirs(path_snap)

from imageio import imsave
from pipeline import SnapMaker, get_data_loader

snap_maker = SnapMaker()
dataloader = get_data_loader(opt)

nb_batch = len(dataloader)
print(nb_batch)

from network import MultiPatchDiscriminator, ResidualGenerator, weights_init, Loss

network_D = MultiPatchDiscriminator(opt).apply(weights_init)
network_G = ResidualGenerator(opt).apply(weights_init)

if ngpu > 1 :
    network_D = nn.DataParallel(network_D)
    network_G = nn.DataParallel(network_G)

network_D = network_D.to(device)
network_G = network_G.to(device)

optim_D = torch.optim.Adam(network_D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
optim_G = torch.optim.Adam(network_G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optim_D, step_size=opt.step_size, gamma=opt.gamma)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optim_G, step_size=opt.step_size, gamma=opt.gamma)

criterion = Loss(opt, device)

epoch = 0
while epoch < opt.epoch_max :

    iter_ = 0
    t0 = time.time()
    losses_D = []
    losses_G = []
    losses_F = []

    for _, pair in enumerate(dataloader):
        inp, tar = pair
        if cuda == True :
            inp = inp.to(device)
            tar = tar.to(device)

        gen, loss_D, loss_G_fake, loss_G_FM = criterion(network_D, network_G, inp, tar)
        loss_G = loss_G_fake + loss_G_FM

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        iter_ += 1

        losses_D.append(loss_D.item())
        losses_G.append(loss_G_fake.item())
        losses_F.append(loss_G_FM.item()) 

        if iter_ % opt.display_frequency == 0 :
            loss_D_mean = np.mean(losses_D)
            loss_G_mean = np.mean(losses_G)
            loss_F_mean = np.mean(losses_F)

            pallete = '[%d/%d][%d/%d] Loss_D: %5.3f Loss_G: %5.3f Loss_F: %5.3f Time: %dsec'
            values = (epoch, opt.epoch_max, iter_, nb_batch, loss_D_mean, loss_G_mean, loss_F_mean, time.time() - t0)
            print(pallete % values)

            if ngpu > 1 :
                state = {'network_G':network_G.module.state_dict(),
                        'network_D':network_D.module.state_dict(),
                        'optimizer_G': optim_G.state_dict(),
                        'optimizer_D': optim_D.state_dict(),
                        'scheduler_G': scheduler_G.state_dict(),
                        'scheduler_D': scheduler_D.state_dict()}
            else :
                state = {'network_G':network_G.state_dict(),
                        'network_D':network_D.state_dict(),
                        'optimizer_G': optim_G.state_dict(),
                        'optimizer_D': optim_D.state_dict(),
                        'scheduler_G': scheduler_G.state_dict(),
                        'scheduler_D': scheduler_D.state_dict()}

            torch.save(state, '%s/%s.last.pt'%(path_model, opt.prefix))

            network_G.eval()

            snap_inp = np.hstack([inp[n].detach().cpu().numpy().squeeze(0) for n in range(4)])
            snap_tar = np.hstack([tar[n].detach().cpu().numpy().squeeze(0) for n in range(4)])
            snap_gen = np.hstack([gen[n].detach().cpu().numpy().squeeze(0) for n in range(4)])

            snap = np.vstack([snap_inp, snap_tar, snap_gen])
            snap = snap_maker(snap)

            imsave('%s/%s.%04d.%07d.png'%(path_snap, opt.prefix, epoch, iter_), snap)

            t0 = time.time()
            losses_D = []
            losses_G = []
            losses_F = []

            network_G.train()

    epoch += 1
    scheduler_G.step()
    scheduler_D.step()

    if ngpu > 1 :
        state = {'network_G':network_G.module.state_dict(),
                'network_D':network_D.module.state_dict(),
                'optimizer_G': optim_G.state_dict(),
                'optimizer_D': optim_D.state_dict(),
                'scheduler_G': scheduler_G.state_dict(),
                'scheduler_D': scheduler_D.state_dict()}
    else :
        state = {'network_G':network_G.state_dict(),
                'network_D':network_D.state_dict(),
                'optimizer_G': optim_G.state_dict(),
                'optimizer_D': optim_D.state_dict(),
                'scheduler_G': scheduler_G.state_dict(),
                'scheduler_D': scheduler_D.state_dict()}

    torch.save(state, '%s/%s.%04d.pt'%(path_model, opt.prefix, epoch))









