import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from pytorch_msssim import ssim
from models.mods import HFRM
from utils.utils import *
from utils.FDL import *


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, gt_img, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = gt_img.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}

        input_img = x[:, :64, :, :]  #取出输入图片
        n, c, h, w = input_img.shape  #输入图片的size参数
        gt_img = x[:, 64:, :, :]


        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_img.shape[0] // 2 + 1,)).to(self.device) #time_step
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_img.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(gt_img)

        if self.training:
            gt_img = x[:, 64:, :, :]


            x = gt_img * a.sqrt() + e * (1.0 - a).sqrt()

            noise_output = self.Unet(torch.cat([input_img, x], dim=1), t.float())
            denoise_image = self.sample_training(input_img, b, gt_img)


            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = denoise_image
            data_dict["e"] = e

        else:
            denoise_image = self.sample_training(input_img, b,gt_img)


            data_dict["pred_x"] = denoise_image

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.FDL_loss = FDL(loss_weight=1.0,alpha=2.0,patch_factor=4,ave_spectrum=True,log_matrix=True,batch_matrix=True).to(self.device)


        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        self.train_psnr_list = []
        self.test_psnr_list = []
        self.epochs = []

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def plot_psnr(self):
        plt.figure()
        plt.plot(range(1, len(self.train_psnr_list) + 1), self.train_psnr_list, label='Train PSNR')
        plt.plot(self.epochs, self.test_psnr_list, label='Test PSNR', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.title('Training and Testing PSNR Over Epochs')
        plt.savefig('./psnr_plot.png')
        plt.show()



    def validate(self, val_loader):
        self.model.eval()
        total_psnr = 0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                y = y.flatten(start_dim=0, end_dim=1) if y.ndim == 5 else y
                x, y = x.to(self.device), y.to(self.device)
                input_x = torch.cat([x, y], dim=1)
                output = self.model(input_x)
                pred_x = torch.clamp(output["pred_x"], 0., 1.)
                total_psnr += batch_PSNR(pred_x,y, 1.)
                count += 1
        return total_psnr / count

    def train(self, train_data,test_data):
        cudnn.benchmark = True
        train_loader, val_loader = train_data,test_data

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)


        best_psnr=0

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            #self.model.train()
            total_psnr = 0
            count = 0
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                y = y.flatten(start_dim=0, end_dim=1) if y.ndim == 5 else y
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                y = y.to(self.device)
                #print(x.shape)
                #print(y.shape)

                input_x= torch.cat([x,y], dim=1)

                output = self.model(input_x)

                FDLloss, noise_loss, photo_loss = self.estimation_loss(input_x, output)

                loss = noise_loss + 1000*photo_loss + 10*FDLloss
                out_train= torch.clamp(output["pred_x"], 0., 1.) 
                psnr_train = batch_PSNR(out_train,y, 1.)
                total_psnr += psnr_train
                count += 1
                if self.step % 10 == 0:
                    print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, FDL_loss:{:.4f}, "
                          " PSNR: {:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
                                                         noise_loss.item(), 1000*photo_loss.item(),
                                                        10*FDLloss.item(), psnr_train))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step,epoch)

                if psnr_train>best_psnr:
                    best_psnr=psnr_train
                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                'state_dict': self.model.state_dict(),
                                                'optimizer': self.optimizer.state_dict(),
                                                'scheduler': self.scheduler.state_dict(),
                                                'ema_helper': self.ema_helper.state_dict(),
                                                'params': self.args,
                                                'config': self.config},
                                                filename=os.path.join(self.config.data.ckpt_dir, str(epoch)))
                    print("A Best Model Saved!")
            avg_train_psnr = total_psnr / count
            self.train_psnr_list.append(avg_train_psnr)
            if epoch % 10 == 0:
                avg_test_psnr = self.validate(val_loader)
                self.test_psnr_list.append(avg_test_psnr)
                self.epochs.append(epoch)
                print(f'Validation PSNR: {avg_test_psnr:.4f}')
                        
            self.scheduler.step()
        self.plot_psnr()

    def estimation_loss(self, x, output):

        pred_x, noise_output, e = output["pred_x"],output["noise_output"], output["e"]

        gt_img = x[:, 64:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = content_loss + ssim_loss
        #photo_loss = content_loss

        FDLloss=self.FDL_loss(pred_x, gt_img)

        #return noise_loss, photo_loss, frequency_loss
        return noise_loss, photo_loss, FDLloss

    def sample_validation_patches(self, val_loader, step,epoch):
        image_folder = "./results/"
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):

                b, _, img_h, img_w = x.shape
                input_x= torch.cat([x,y], dim=1)

                out = self.model(input_x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(epoch),f"{i}.png"))
                utils.logging.save_image(y[:, :, :img_h, :img_w].detach().cpu(), os.path.join(image_folder, str(epoch),f"GT+{i}.png"))