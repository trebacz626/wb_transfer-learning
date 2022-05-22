import torch
import itertools
from torch.optim import lr_scheduler

from losses.DiceLoss import DiceLoss
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .unet import UNet


class MutualModel(BaseModel):
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_T', 'G_T', 'cycle_T','syn_sup', 'real_sup', 'kd_r_s', 'kd_s_r']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = []
        visual_names_T = ['real_T']
        visual_names_S = []

        if self.isTrain:
            visual_names_A += ['real_A', 'fake_T', 'rec_A']
            visual_names_T += ['fake_A', 'rec_T']
        else:
            visual_names_S += ['p_T_combined']

        self.visual_names = visual_names_A + visual_names_T
        if self.isTrain:
            self.visual_names += ['label_A_show', 'p_A_T_real_show', 'p_A_T_syn_show']
        self.visual_names += ['label_T_show', 'p_T_real_show', 'p_T_syn_show']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_T', 'D_A', 'D_T', 'S_real', 'S_syn']
        else:  # during test time, only load Gs
            self.model_names = ['S_real', 'S_syn']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netG_T = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_T = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netS_real = UNet()

        self.netS_real.to(self.gpu_ids[0])
        self.netS_real = torch.nn.DataParallel(self.netS_real, self.gpu_ids)

        self.netS_syn = UNet()

        self.netS_syn.to(self.gpu_ids[0])
        self.netS_syn = torch.nn.DataParallel(self.netS_syn, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_T_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterion_dice = DiceLoss().to(self.device)
            self.softmax = torch.nn.Softmax(dim=1)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_T = torch.optim.Adam(self.netG_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_T = torch.optim.Adam(self.netD_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_S = torch.optim.Adam(itertools.chain(self.netS_real.parameters(), self.netS_syn.parameters()), lr=opt.lr, betas=(0.9, 0.999))

            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_T)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_T)
            self.optimizers.append(self.optimizer_S)

            self.optimizer_names = ["optimizer_G_A", "optimizer_G_T", "optimizer_D_A", "optimizer_D_T", "optimizer_S"]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_T = input["T_scan"].to(self.device)
        self.label_T = input["T_labels"].to(self.device)
        self.label_T_show = torch.unsqueeze(torch.argmax(input["T_labels"], dim=1)/3.5 - 1, dim=1)

        if self.isTrain:
           self.real_A = input["A_scan"].to(self.device)
           self.label_A = input["A_labels"].to(self.device)
           self.label_A_show = torch.unsqueeze(torch.argmax(input["A_labels"], dim=1)/3.5 - 1, dim=1)

    def get_schedulers(self, opt):
        return [lr_scheduler.StepLR(self.optimizer_S, step_size=2, gamma=0.9)]

    def forward(self):
        torch.autograd.set_detect_anomaly(True)#TODO remove
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if self.isTrain:
            self.fake_T = self.netG_A(self.real_A)
            self.rec_A = self.netG_T(self.fake_T)
            self.fake_A = self.netG_T(self.real_T)
            self.rec_T = self.netG_A(self.fake_A)
            self.p_A_T_real = self.netS_real(self.fake_T)
            self.p_A_T_real_show = torch.unsqueeze(torch.argmax(self.p_A_T_real.cpu(), dim=1)/3.5 - 1, dim=1)
            self.p_A_T_syn = self.netS_syn(self.fake_T)
            self.p_A_T_syn_show = torch.unsqueeze(torch.argmax(self.p_A_T_syn.cpu(), dim=1)/3.5 - 1, dim=1)

        self.p_T_real = self.netS_real(self.real_T)
        self.p_T_real_show = torch.unsqueeze(torch.argmax(self.p_T_real.cpu(), dim=1)/3.5 - 1, dim=1)

        self.p_T_syn = self.netS_syn(self.real_T)
        self.p_T_syn_show = torch.unsqueeze(torch.argmax(self.p_T_syn.cpu(), dim=1)/3.5 - 1, dim=1)

        if not self.isTrain:
            self.p_T_combined = (self.softmax(self.p_T_real) + self.softmax(self.p_T_syn))/2

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        torch.set_printoptions(profile="full")
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.optimizer_D_A.zero_grad()
        fake_T = self.fake_T_pool.query(self.fake_T)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_T, fake_T)
        self.optimizer_D_A.step()

    def backward_D_T(self):
        """Calculate GAN loss for discriminator D_T"""
        self.optimizer_D_T.zero_grad()
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_T = self.backward_D_basic(self.netD_T, self.real_A, fake_A)
        self.optimizer_D_T.step()

    def backward_G_A_T(self):
        lambda_A = self.opt.lambda_A
        lambda_T = self.opt.lambda_T
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_T), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
        self.loss_cycle_T = self.criterionCycle(self.rec_T, self.real_T)
        self.loss_cycle = self.loss_cycle_A * lambda_A + self.loss_cycle_T * lambda_T
        self.loss_cycle.backward(retain_graph=True)
        self.loss_syn_sup = self.criterionCE(self.p_A_T_syn, self.label_A) + self.criterion_dice(self.p_A_T_syn, self.label_A)

        self.loss_G_A_T = self.loss_G_A + self.loss_syn_sup#+ self.loss_cycle
        self.loss_G_A_T.backward(retain_graph=True)
        return self.loss_G_A_T

    def backward_G_T_A(self):

        self.loss_G_T = self.criterionGAN(self.netD_T(self.fake_A), True)

        self.loss_G_T_A = self.loss_G_T #+ self.loss_cycle
        self.loss_G_T_A.backward()
        return self.loss_G_T_A

    def backward_S(self):
        lambda_KD_1 = self.opt.lambda_KD_1
        lambda_KD_2 = self.opt.lambda_KD_2
        self.optimizer_S.zero_grad()


        self.loss_real_sup = self.criterionCE(self.p_T_real, self.label_T) + self.criterion_dice(self.p_T_real, self.label_T)
        self.loss_kd_s_r = self.criterionCE(self.p_A_T_real, self.softmax(self.p_A_T_syn))
        self.loss_real_seg = self.loss_kd_s_r * lambda_KD_1 + self.loss_real_sup

        # self.loss_syn_sup = self.criterionCE(self.p_A_T_syn, self.label_A) + self.criterion_dice(self.p_A_T_syn, self.label_A)
        self.loss_kd_r_s = self.criterionCE(self.p_T_syn, self.softmax(self.p_T_real))
        self.loss_syn_seg = self.loss_kd_r_s * lambda_KD_2 + self.loss_syn_sup

        self.loss_real_seg.backward(retain_graph=True)
        self.loss_syn_seg.backward()
        # self.optimizer_S.step()
        return self.loss_syn_seg, self.loss_real_seg

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_G_T.zero_grad()
        self.optimizer_G_A.zero_grad()
        self.optimizer_S.zero_grad()
        self.set_requires_grad([self.netD_A, self.netD_T], False)
        self.backward_G_A_T()
        self.set_requires_grad([self.netD_T], True)
        self.backward_D_T()
        self.set_requires_grad([self.netD_T], False)
        self.backward_G_T_A()
        self.set_requires_grad([self.netD_A], True)
        self.backward_D_A()
        self.set_requires_grad([ self.netD_A], False)
        self.set_requires_grad([self.netG_A, self.netG_T], False)
        self.backward_S()
        self.set_requires_grad([self.netG_A, self.netG_T], True)
        self.optimizer_S.step()
        self.optimizer_G_A.step()
        self.optimizer_G_T.step()

    def optimize_parameters_simple(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

        self.optimizer_G_A.zero_grad()
        self.optimizer_G_T.zero_grad()
        self.optimizer_S.zero_grad()
        self.set_requires_grad([self.netD_A, self.netD_T], False)
        self.backward_G_A_T()
        self.backward_G_T_A()
        self.backward_S()
        self.optimizer_S.step()
        self.optimizer_G_A.step()
        self.optimizer_G_T.step()
        self.set_requires_grad([self.netD_T, self.netD_A], True)
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_T.zero_grad()
        self.backward_D_T()
        self.backward_D_A()
        self.optimizer_D_A.step()
        self.optimizer_D_T.step()

    def validation_scores(self):
        self.forward()
        return {
            "valid_dice_score": 1 - self.criterion_dice(self.p_T_combined, self.label_T, softmax=False),
            "valid_dice_score_hard": 1 - self.criterion_dice(self.p_T_combined, self.label_T, softmax=False, one_hot_encode=True)

        }
