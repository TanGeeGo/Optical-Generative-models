import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from PIL import Image
from diffusers.utils import make_image_grid
from dataclasses import dataclass
from tqdm.auto import tqdm
from initialization import extract_material_parameter

class Generator_ClsEmd_light(nn.Module):
    def __init__(self, img_size, in_channel=1, num_classes=10, dim_expand_ratio=8):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.dim_expand_ratio = dim_expand_ratio

        self.class_embedding = nn.Embedding(num_classes, dim_expand_ratio)

        # # preprocessing, start from img_size // 4 size 
        self.l1 = nn.Sequential(nn.Linear(img_size*img_size+dim_expand_ratio, (img_size)*(img_size)),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size, (img_size)*(img_size)),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size, (img_size)*(img_size)+1))

    def forward(self, x, classes):
        b, _, _, _ = x.shape
        x_flat = torch.cat((x.view(b, -1).contiguous(), self.class_embedding(classes)), -1)
        out = self.l1(x_flat)
        out_img = out[:, :-1]
        out_scale = out[:, -1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        img = out_img.view(out.shape[0], self.in_channel, self.img_size, self.img_size).contiguous()
        return img, out_scale

class FreeSpaceProp(nn.Module):
    def __init__(self, 
                 wlength_vc, 
                 ridx_air,
                 total_x_num, total_y_num,
                 dx, dy,
                 prop_z):
        super(FreeSpaceProp, self).__init__()
        self.prop_z = prop_z
        self.total_x_num = total_x_num
        self.total_y_num = total_y_num
        self.dx = dx
        self.dy = dy

        wlengtheff = wlength_vc / ridx_air # effect wavelength
        
        y, x = (dy * float(total_y_num), dx * float(total_x_num))

        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), total_y_num)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), total_x_num)

        fx, fy = torch.meshgrid(fx, fy, indexing='ij')

        f0 = 1 / wlengtheff # to ensure coherent? It is the diffraction limitation.
        fy_max = 1 / np.sqrt((2 * prop_z * (1 / y))**2 + 1) / wlengtheff
        fx_max = 1 / np.sqrt((2 * prop_z * (1 / x))**2 + 1) / wlengtheff
        Q = torch.tensor(((np.abs(np.array(fx)) < fx_max) & (np.abs(np.array(fy)) < fy_max)).astype(np.uint8), 
                         dtype=torch.float32)

        # Angle Spectrum
        prop_window = Q * (fx**2 + fy**2) * (wlengtheff**2)
        phase_change = 2 * np.pi * f0 * prop_z * torch.sqrt(torch.clamp(1 - prop_window, 
                                                                        min=0, max=1))
        phase_change_cplx = torch.complex(torch.cos(phase_change),
                                          torch.sin(phase_change)) * Q # as exp(1i*...)
        
        shifted_phase_change_cplx = torch.fft.ifftshift(phase_change_cplx)
        shifted_phase_change_cplx = shifted_phase_change_cplx[None, None, ...]

        self.register_buffer('shifted_phase_change_cplx',
                             shifted_phase_change_cplx)
        
        
    def forward(self, x):
        
        ASpectrum = torch.fft.fft2(x)
        ASpectrum_z = torch.mul(self.shifted_phase_change_cplx, ASpectrum)
        output = torch.fft.ifft2(ASpectrum_z)
        return output
    
    def update_proplocation(self, x_shift, y_shift, z_shift):
        
        self.prop_z = self.prop_z + z_shift.item()
        Wz =  self.f0 * self.prop_z * torch.sqrt(torch.clamp(1 - self.prop_window, min=0, max=1))
        
        phase_change = 2 * np.pi * (Wz + self.fx * x_shift.item() + self.fy * y_shift.item())
        phase_change_cplx = torch.complex(torch.cos(phase_change),
                                          torch.sin(phase_change)) * self.Q
        
        # band pass for off axis numerical propagation
        phase_change_cplx = self._bandpass(phase_change_cplx, self.fx, self.fy, 
                                           self.total_x_num*self.dx, self.total_y_num*self.dy,
                                           x_shift.item(), y_shift.item(), self.prop_z, self.wlengtheff)

        shifted_phase_change_cplx = torch.fft.ifftshift(phase_change_cplx)
        shifted_phase_change_cplx = shifted_phase_change_cplx[None, None, ...].to(z_shift.device)

        self.register_buffer('shifted_phase_change_cplx',
                             shifted_phase_change_cplx)
        
    def _bandpass(self, H, fX, fY, Sx, Sy, x0, y0, z0, wv):
        """
        Table 1 of "Shifted angular spectrum method for off-axis numerical propagation" (2010).
        :param Sx:
        :param Sy:
        :param x0:
        :param y0:
        :return:
        """

        du = 1 / (2 * Sx)
        u_limit_p = ((x0 + 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
        u_limit_n = ((x0 - 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv

        if Sx < x0:
            u0 = (u_limit_p + u_limit_n) / 2
            u_width = u_limit_p - u_limit_n
        elif x0 <= -Sx:
            u0 = -(u_limit_p + u_limit_n) / 2
            u_width = u_limit_n - u_limit_p
        else:
            u0 = (u_limit_p - u_limit_n) / 2
            u_width = u_limit_p + u_limit_n

        dv = 1 / (2 * Sy)
        v_limit_p = ((y0 + 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
        v_limit_n = ((y0 - 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv

        if Sy < y0:
            # print('Sy < y0')
            v0 = (v_limit_p + v_limit_n) / 2
            v_width = v_limit_p - v_limit_n
        elif y0 <= -Sy:
            # print('y0 <= -Sy')
            v0 = -(v_limit_p + v_limit_n) / 2
            v_width = v_limit_n - v_limit_p
        else:
            # print('else')
            v0 = (v_limit_p - v_limit_n) / 2
            v_width = v_limit_p + v_limit_n

        fx_max = u_width / 2
        fy_max = v_width / 2

        H_filter = (torch.abs(fX - u0) <= fx_max) * (torch.abs(fY - v0) < fy_max)

        H_final = H * H_filter

        return H_final
        
class MaskBlockPhase(nn.Module):
    def __init__(self, 
                 mask_x_num, mask_y_num, mask_base_thick,
                 mask_init_method,
                 total_x_num, total_y_num,
                 ridx_mask, freq, c, attenu_factor):
        super(MaskBlockPhase, self).__init__()
        self.register_parameter('mask_phase',
                                nn.Parameter(torch.Tensor(1, 1, mask_x_num, mask_y_num),
                                             requires_grad=True))
        
        if mask_init_method == 'zero':
            nn.init.zeros_(self.mask_phase.data)
        elif mask_init_method == 'normal':
            nn.init.normal_(self.mask_phase.data, mean=0., std=0.5)

        self.pad_x = total_x_num // 2 - mask_x_num // 2
        self.pad_y = total_y_num // 2 - mask_y_num // 2 # needs to pad zero
        
        self.phase_change_factor = 2 * np.pi * (ridx_mask - 1) \
            * freq / c # (n-1)*k used to calculate the height from phase
        self.amp_decay_factor = 2 * np.pi * attenu_factor \
            * freq / c # amplitude decay
        
        self.mask_base_thick = mask_base_thick

    def forward(self, x):
        # sigmoid on the mask phase, to prevent out of [0, 2pi] phase
        mask_phase = torch.sigmoid(self.mask_phase) * 2 * np.pi
        mask_amp = torch.ones_like(mask_phase)

        mask_phase = F.pad(mask_phase, (self.pad_y, self.pad_y, self.pad_x, self.pad_x))
        mask_amp = F.pad(mask_amp, (self.pad_y, self.pad_y, self.pad_x, self.pad_x))

        # NOTE: it is the phase of the mask, not the height
        mask_cplx = torch.complex(torch.cos(mask_phase),
                                  torch.sin(mask_phase)) * torch.abs(mask_amp)
        
        # if amplitude decay
        if self.amp_decay_factor > 0.:
            mask_height = self.mask_base_thick + \
                mask_phase / self.phase_change_factor
            mask_amp = torch.exp(- mask_height * self.amp_decay_factor)
            mask_cplx = mask_amp * mask_cplx

        output = torch.mul(mask_cplx, x) # should be mul not matmul
        return output

class D2nnModel(nn.Module):
    def __init__(self, 
                 img_size, 
                 in_channel, 
                 c, freq,
                 num_masks,
                 wlength_vc,
                 ridx_air, ridx_mask, attenu_factor,
                 total_x_num, total_y_num,
                 mask_x_num, mask_y_num, mask_init_method, mask_base_thick,
                 dx, dy,
                 object_mask_dist, mask_mask_dist, mask_sensor_dist,
                 obj_x_num, obj_y_num,):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channel
        self.total_x_num = total_x_num
        self.total_y_num = total_y_num
        self.mask_x_num = mask_x_num
        self.mask_y_num = mask_y_num
        self.obj_x_num = obj_x_num
        self.obj_y_num = obj_y_num

        # D2nn blocks
        self.D2NN = nn.ModuleList()
        self.D2NN.append(FreeSpaceProp(wlength_vc=wlength_vc, 
                                        ridx_air=ridx_air,
                                        total_x_num=total_x_num, total_y_num=total_y_num,
                                        dx=dx, dy=dy,
                                        prop_z=object_mask_dist)) # distance takein here
        self.D2NN.append(MaskBlockPhase(mask_x_num=mask_x_num, mask_y_num=mask_y_num, 
                                        mask_base_thick=mask_base_thick, mask_init_method=mask_init_method,
                                        total_x_num=total_x_num, total_y_num=total_y_num, ridx_mask=ridx_mask,
                                        freq=freq, c=c, attenu_factor=attenu_factor))
        
        for _ in range(num_masks - 1):
            self.D2NN.append(FreeSpaceProp(wlength_vc=wlength_vc, 
                                            ridx_air=ridx_air,
                                            total_x_num=total_x_num, total_y_num=total_y_num,
                                            dx=dx, dy=dy,
                                            prop_z=mask_mask_dist)) 
            self.D2NN.append(MaskBlockPhase(mask_x_num=mask_x_num, mask_y_num=mask_y_num, 
                                            mask_base_thick=mask_base_thick, mask_init_method=mask_init_method,
                                            total_x_num=total_x_num, total_y_num=total_y_num, ridx_mask=ridx_mask,
                                            freq=freq, c=c, attenu_factor=attenu_factor))
            
        self.D2NN.append(FreeSpaceProp(wlength_vc=wlength_vc, 
                                        ridx_air=ridx_air,
                                        total_x_num=total_x_num, total_y_num=total_y_num,
                                        dx=dx, dy=dy,
                                        prop_z=mask_sensor_dist))
        
    def forward(self, x):
        img_cplx = self.img_preprocess(x) # change to the phase on SLM

        for blocks in self.D2NN:
            img_cplx = blocks(img_cplx)

        # post-processing the image
        img_cplx = self.center_crop(img_cplx, [self.obj_y_num, self.obj_x_num])

        img_cplx = torch.abs(img_cplx) # complex amplitude to intensity
        
        output = F.avg_pool2d(img_cplx, kernel_size=self.obj_y_num//self.img_size, 
                              stride=self.obj_y_num//self.img_size, padding=0)
        
        output = torch.square(output)

        return output

    def img_preprocess(self, x):
        alpha = 1.0
        x = (x * alpha * np.pi + alpha * np.pi).clamp(0.0, 2 * alpha * torch.tensor(np.pi).to(x.device))
        x_cplx = torch.complex(torch.cos(x), torch.sin(x))
        img_input = self.resize_phase_complex(x_cplx)
        return img_input
    
    def resize_phase_complex(self, x):
        pad_x = (self.total_x_num // 2 - self.obj_x_num // 2)
        pad_y = (self.total_y_num // 2 - self.obj_y_num // 2)
        output_real = F.interpolate(x.real, size=[self.obj_y_num, self.obj_x_num], mode='nearest')
        output_imag = F.interpolate(x.imag, size=[self.obj_y_num, self.obj_x_num], mode='nearest')
        output_real = F.pad(output_real, (pad_y, pad_y, pad_x, pad_x))
        output_imag = F.pad(output_imag, (pad_y, pad_y, pad_x, pad_x))
        
        return torch.complex(output_real, output_imag)
    
    def center_crop(self, x: torch.tensor, size: list):
        output = x[..., self.total_y_num // 2 - size[0] // 2 : self.total_y_num // 2 + size[0] // 2,
                        self.total_x_num // 2 - size[1] // 2 : self.total_x_num // 2 + size[1] // 2]
        output = output.contiguous()
        return output

@dataclass
class TestConfig:
    num_epochs = 1
    save_model_epochs = 10

    seed = 0
    img_size = 32
    in_channel = 1
    test_batch_size = 36

    lr_d2nn = 2e-4
    lr_deepmodel = 2e-4

    # D2NN parameters
    c = 299792458 
    ridx_air = 1.0
    wlength_vc = 520e-9
    freq = c / wlength_vc 
    amp_modulation = False                   # Consider or not the absorption of the masks
    ridx_mask, attenu_factor = extract_material_parameter(freq, amp_modulation)
    num_masks = 1
    object_mask_dist, mask_mask_dist, mask_sensor_dist = 12.01e-02, 2.0e-02, 9.64e-02
    mask_base_thick = 1.0e-03
    total_x_num, total_y_num =  800, 800     # number of pixels on the mask
    mask_x_num, mask_y_num = 400, 400        # number of trainable pixels
    dx, dy = 8e-06, 8e-06                    # pixel size of the mask, about 
    obj_x_num, obj_y_num = 320, 320
    seed = 0
    mask_init_method = 'zero'

    output_dir = './test'

    load_pretrain = True
    ckpt_to_load = './ckpt/ckpt_mnist.pth'
    
def main():
    config = TestConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)

    # define the generative model
    generator_e = Generator_ClsEmd_light(img_size=config.img_size, in_channel=config.in_channel,
                                         num_classes=10, dim_expand_ratio=128)
    generator_d = D2nnModel(img_size=config.img_size,
                            in_channel=config.in_channel,
                            c=config.c, freq=config.freq, num_masks=config.num_masks,
                            wlength_vc=config.wlength_vc,
                            ridx_air=config.ridx_air, ridx_mask=config.ridx_mask,
                            attenu_factor=config.attenu_factor,
                            total_x_num=config.total_x_num, total_y_num=config.total_y_num,
                            mask_x_num=config.mask_x_num, mask_y_num=config.mask_y_num,
                            mask_init_method=config.mask_init_method, mask_base_thick=config.mask_base_thick,
                            dx=config.dx, dy=config.dy, 
                            object_mask_dist=config.object_mask_dist, 
                            mask_mask_dist=config.mask_mask_dist,
                            mask_sensor_dist=config.mask_sensor_dist, 
                            obj_x_num=config.obj_x_num, obj_y_num=config.obj_y_num)
    
    if config.load_pretrain:
        checkpoint = torch.load(config.ckpt_to_load, weights_only=True)
        generator_e.load_state_dict(checkpoint['ge_model_state_dict'])
        generator_d.load_state_dict(checkpoint['gd_model_state_dict'])

    generator_e = generator_e.to('cuda')
    generator_d = generator_d.to('cuda')

    with torch.no_grad():
        noise = torch.randn(config.test_batch_size, config.in_channel, 
                            config.img_size, config.img_size, 
                            generator=torch.manual_seed(config.seed)).to('cuda')
        labels = torch.randint(0, 10, size=(config.test_batch_size, 1), 
                               dtype=torch.int64, generator=torch.manual_seed(config.seed)).squeeze().to('cuda')

        gen_img, _ = generator_e(noise, labels)
        d2nn_img = generator_d(gen_img)

        d2nn_img_min = torch.min(torch.min(d2nn_img, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        d2nn_img_max = torch.max(torch.max(d2nn_img, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        d2nn_img = (d2nn_img - d2nn_img_min) / (d2nn_img_max - d2nn_img_min) 
        
        d2nn_img = (d2nn_img * 255).byte().cpu().permute(0, 2, 3, 1).numpy()

        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in d2nn_img]

        image_grid = make_image_grid(pil_images, rows=int(np.sqrt(config.test_batch_size)), 
                                                 cols=int(np.sqrt(config.test_batch_size)))
        image_grid.save(os.path.join(config.output_dir, 'example_mnist.png'))

if __name__ == "__main__":
    main()