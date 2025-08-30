import os
import json
import math
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from safetensors.torch import load_file
from tqdm.auto import tqdm
from torchvision import transforms
from datasets import load_from_disk
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher
from functools import partial

from utils import _extract_into_tensor, kl_divergence_loss
from models import Snapshot_Optical_Generative_Model, Multicolor_Optical_Generative_Model, Iterative_Optical_Generative_Model
from pipeline_costum import DDPMPipeline_Costum, DDPMPipeline_Costum_ClsEmb

def parse_args():
    parser = argparse.ArgumentParser(description="Main configuration for optical generative models")
    parser.add_argument("--task", type=str, default="diffusion_digital", help="training task", 
                        choices=["diffusion_digital", "snapshot_optical", "multicolor_optical", "iterative_optical"])
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPU to use")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./logs/exp", help="Where to save logs and checkpoints")
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--train_batch_size", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate_digital", type=float, default=1e-4)
    parser.add_argument("--learning_rate_optical", type=float, default=5e-3)
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"])
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default='linear')
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"])
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument("--save_image_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=50)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=0, help="For the dataset with class indices")

    ## for diffusion_digital task
    parser.add_argument("--time_embedding_type_d", type=str, default="positional")
    parser.add_argument(
        "--down_block_types",
        type=str,
        nargs="+",
        default=["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"]
    )
    parser.add_argument(
        "--up_block_types",
        type=str,
        nargs="+",
        default=["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"]
    )
    parser.add_argument(
        "--block_out_channels",
        type=int,
        nargs="+",
        default=[224, 448, 672, 896]
    )
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--prediction_type_d", type=str, default='epsilon')

    ## for optical generation task
    parser.add_argument("--c", type=float, default=299792458.0)
    parser.add_argument("--ridx_air", type=float, default=1.0)
    
    parser.add_argument("--object_layer_dist", type=float, default=5e-2)
    parser.add_argument("--layer_layer_dist", type=float, default=1e-2)
    parser.add_argument("--layer_sensor_dist", type=float, default=5e-2)

    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--total_num", type=int, default=800)
    parser.add_argument("--obj_num", type=int, default=320)
    parser.add_argument("--layer_neuron_num", type=int, default=400)
    parser.add_argument("--dxdy", type=float, default=8e-6)
    parser.add_argument("--layer_init_method", type=str, default='zero')

    parser.add_argument("--amp_modulation", type=bool, default=False)

    ## for snapshot_optical task
    parser.add_argument("--teacher_ckpt_snst", type=str, default="./teacher_snapshot")

    parser.add_argument("--inference_acc_snst", type=bool, default=True)
    parser.add_argument("--acc_ratio_snst", type=int, default=20)
    
    parser.add_argument("--wavelength_snst", type=float, default=520e-9)
    parser.add_argument("--ridx_layer_snst", type=float, default=1.0) # needs update for physical layer
    parser.add_argument("--attenu_factor_snst", type=float, default=0.0) # needs update for physical layer

    parser.add_argument("--apply_scale_snst", type=bool, default=True)
    parser.add_argument("--scale_type_snst", type=str, default="neural_pred", choices=["static_mean", "neural_pred"])
    parser.add_argument("--noise_perturb_snst", type=float, default=1e-4)
    parser.add_argument("--eval_kl_snst", type=bool, default=False)
    parser.add_argument("--kl_ratio_snst", type=float, default=1e-4)

    ## for multicolor_optical task
    parser.add_argument("--teacher_ckpt_mtcl", type=str, default="./teacher_multicolor")

    parser.add_argument("--inference_acc_mtcl", type=bool, default=True)
    parser.add_argument("--acc_ratio_mtcl", type=int, default=20)
    
    parser.add_argument("--wavelength_mtcl", type=float, nargs='+', default=[450e-9, 520e-9, 638e-9], help="List of wavelengths in meters.")
    parser.add_argument("--ridx_layer_mtcl", type=float, nargs='+', default=[1.0, 1.0, 1.0]) # needs update for physical layer
    parser.add_argument("--attenu_factor_mtcl", type=float, nargs='+', default=[0.0, 0.0, 0.0]) # needs update for physical layer

    parser.add_argument("--apply_scale_mtcl", type=bool, default=False)
    parser.add_argument("--scale_type_mtcl", type=str, default="static_mean", choices=["static_mean", "neural_pred"])
    parser.add_argument("--noise_perturb_mtcl", type=float, default=1e-4)
    parser.add_argument("--eval_kl_mtcl", type=bool, default=False)
    parser.add_argument("--kl_ratio_mtcl", type=float, default=1e-4)

    # for iterative_optical task
    parser.add_argument("--time_embedding_type_o", type=str, default="positional")

    parser.add_argument("--wavelength_itrt", type=float, nargs='+', default=[450e-9, 520e-9, 638e-9], help="List of wavelengths in meters.")
    parser.add_argument("--ridx_layer_itrt", type=float, nargs='+', default=[1.0, 1.0, 1.0]) # needs update for physical layer
    parser.add_argument("--attenu_factor_itrt", type=float, nargs='+', default=[0.0, 0.0, 0.0]) # needs update for physical layer

    parser.add_argument("--prediction_type_o", type=str, default='sample', choices=["epsilon", "sample"])
    parser.add_argument("--beta_start_itrt", type=float, default=0.001)
    parser.add_argument("--beta_end_itrt", type=float, default=0.010)

    args = parser.parse_args()
    return args

def main(args):
    if args.output_dir is None:
        os.makedirs(args.output_dir, exist_ok=True)

    # load the dataset
    dataset = load_from_disk(args.data_path)['train']
    preprocess = transforms.Compose([
        transforms.Resize(size=(args.sample_size, args.sample_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def transform(examples, wz_cls=False):
        images = [preprocess(image) for image in examples["image"]]
        if wz_cls:
            labels = [torch.tensor(label) for label in examples["label"]]

            return {"images": images, "labels": labels}
        else:
            return {"images": images}
    
    wz_cls = False if not args.num_classes else True
    transform_fn = partial(transform, wz_cls=wz_cls)
    dataset.set_transform(transform_fn)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.task == 'diffusion_digital':
        train_diffusion(args, train_dataloader, wz_cls=wz_cls)
    elif args.task == 'snapshot_optical':
        train_snapshot(args, train_dataloader, wz_cls=wz_cls)
    elif args.task == 'multicolor_optical':
        train_multicolor(args, train_dataloader, wz_cls=wz_cls)
    elif args.task == 'iterative_optical':
        train_iterative(args, train_dataloader, wz_cls=wz_cls)        

def train_diffusion(args, dataloader, wz_cls):
    if wz_cls:
        model = UNet2DModel(
            sample_size=args.sample_size,  
            in_channels=args.in_channels,       
            out_channels=args.in_channels,   
            time_embedding_type=args.time_embedding_type_d,
            down_block_types=args.down_block_types,
            up_block_types=args.up_block_types,
            block_out_channels=args.block_out_channels,
            layers_per_block=args.layers_per_block,
            num_class_embeds=args.num_classes
        )
    else:
        model = UNet2DModel(
            sample_size=args.sample_size,  
            in_channels=args.in_channels,       
            out_channels=args.in_channels,   
            time_embedding_type=args.time_embedding_type_d,
            down_block_types=args.down_block_types,
            up_block_types=args.up_block_types,
            block_out_channels=args.block_out_channels,
            layers_per_block=args.layers_per_block
        )

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, 
                                    beta_schedule=args.ddpm_beta_schedule,
                                    prediction_type=args.prediction_type_d)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_digital)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs)
    )

    def evaluate(args, epoch, pipeline):
        images = pipeline(
            batch_size=args.eval_batch_size,
            generator=torch.manual_seed(args.seed),
            num_inference_steps=args.ddpm_num_steps
        ).images

        image_grid = make_image_grid(images, 
                                     rows=int(math.sqrt(args.eval_batch_size)), 
                                     cols=int(math.sqrt(args.eval_batch_size)))

        test_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, wz_cls):
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_diffusion")

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        ## training
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                if wz_cls:
                    labels = batch['labels']
                else:
                    labels = None
                
                noise = torch.randn_like(clean_images)
    
                batch_size = clean_images.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                    device=clean_images.device, dtype=torch.int64)
                
                # add noise according to the noise magnitude of each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    if labels is None:
                        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    else:
                        noise_pred = model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]

                    if args.prediction_type_d == "epsilon":    
                        loss = F.mse_loss(noise_pred, noise)
                    elif args.prediction_type_d == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(noise_pred.float(), 
                                                        clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            if accelerator.is_main_process:
                if wz_cls:
                    pipeline = DDPMPipeline_Costum_ClsEmb(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                else:
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
                    evaluate(args, epoch, pipeline)
                
                if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    pipeline.save_pretrained(args.output_dir)

    # start the train loop
    train_loop(args, model, noise_scheduler, optimizer, dataloader, lr_scheduler, wz_cls)
    # start the train loop with notebook launcher (only when you using notebook to launch the training)
    # config = (args, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    # notebook_launcher(train_loop, config, num_processes=args.num_gpu)

def train_snapshot(args, dataloader, wz_cls):
    model = Snapshot_Optical_Generative_Model(
        img_size=args.sample_size,
        in_channel=args.in_channels,
        num_classes=args.num_classes,
        dim_expand_ratio=128,
        c=args.c, num_masks=args.num_layer,
        wlength_vc=args.wavelength_snst,
        ridx_air=args.ridx_air, ridx_mask=args.ridx_layer_snst,
        attenu_factor=args.attenu_factor_snst,
        total_x_num=args.total_num, total_y_num=args.total_num,
        mask_x_num=args.layer_neuron_num, mask_y_num=args.layer_neuron_num,
        mask_init_method=args.layer_init_method, mask_base_thick=1.0e-3,
        dx=args.dxdy, dy=args.dxdy, 
        object_mask_dist=args.object_layer_dist, 
        mask_mask_dist=args.layer_layer_dist,
        mask_sensor_dist=args.layer_sensor_dist, 
        obj_x_num=args.obj_num, obj_y_num=args.obj_num
    )

    # load the pretrained teacher diffusion
    with open(os.path.join(args.teacher_ckpt_snst, 'unet', 'config.json'), 'r') as f:
        unet_config = json.load(f)

    teacher_unet = UNet2DModel(**unet_config)
    teacher_unet_state_dict = load_file(
        os.path.join(args.teacher_ckpt_snst, 'unet', 'diffusion_pytorch_model.safetensors'))
    teacher_unet.load_state_dict(state_dict=teacher_unet_state_dict, strict=True)

    with open(os.path.join(args.teacher_ckpt_snst, 'scheduler', 'scheduler_config.json'), 'r') as f:
        scheduler_config = json.load(f)

    if args.inference_acc_snst:
        noise_scheduler = DDIMScheduler(num_train_timesteps=scheduler_config['num_train_timesteps'],
                                        beta_start=scheduler_config['beta_start'],
                                        beta_end=scheduler_config['beta_end'],
                                        prediction_type=scheduler_config['prediction_type'],
                                        beta_schedule=scheduler_config['beta_schedule'])
    else:
        noise_scheduler = DDPMScheduler(**scheduler_config)

    if wz_cls:
        teacher_pipeline = DDPMPipeline_Costum_ClsEmb(unet=teacher_unet, scheduler=noise_scheduler)
    else:
        teacher_pipeline = DDPMPipeline_Costum(unet=teacher_unet, scheduler=noise_scheduler)

    optimizer = torch.optim.AdamW([{'params': model.DD.parameters(), 'lr': args.learning_rate_optical},
                                   {'params': model.DE.parameters(), 'lr': args.learning_rate_digital}], betas=(0.5, 0.999))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs)
    )

    def evaluate(args, epoch, model, wz_cls, device):
        if wz_cls:
            class_sample = torch.randint(0, args.num_classes, (args.eval_batch_size,), 
                                         generator=torch.manual_seed(args.seed), dtype=torch.int64).to(device)
        else:
            class_sample = None

        noise_sample = torch.randn(args.eval_batch_size, args.in_channels, args.sample_size, args.sample_size, 
                                   generator=torch.manual_seed(args.seed)).to(device=device)

        images, _ = model(noise_sample, class_sample)
        images = images.clip(0, 1)

        img_eval = (images * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
        if args.in_channels == 1:
            pil_img_eval = [Image.fromarray(image.squeeze(), mode='L') for image in img_eval]
        else:
            pil_img_eval = [Image.fromarray(image) for image in img_eval]

        image_grid = make_image_grid(pil_img_eval, 
                                     rows=int(math.sqrt(args.eval_batch_size)), 
                                     cols=int(math.sqrt(args.eval_batch_size)))

        test_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train_loop(args, model, pipeline, optimizer, train_dataloader, lr_scheduler, wz_cls):
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_snapshot")

        model, pipeline.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, pipeline.unet, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        ## training
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                if wz_cls:
                    labels = batch['labels']
                else:
                    labels = None

                with torch.no_grad():
                    if wz_cls:
                        targets, noises, class_sample = \
                            pipeline(batch_size=args.train_batch_size, 
                                    class_sample=labels,
                                    num_inference_steps=int(args.ddpm_num_steps // args.acc_ratio_snst), 
                                    output_type='tensor', return_dict=False)
                    else:
                        targets, noises = \
                            pipeline(batch_size=args.train_batch_size, 
                                    num_inference_steps=int(args.ddpm_num_steps // args.acc_ratio_snst), 
                                    output_type='tensor', return_dict=False)
                        class_sample = None
                    
                    targets = (targets / 2. + 0.5).clamp(0, 1)
                    noises = noises + args.noise_perturb_snst * torch.randn_like(noises) # following stable diffusion, accelerating converging

                with accelerator.accumulate(model):
                    if labels is None:
                        outputs, scale = model(noises)
                    else:
                        outputs, scale = model(noises, labels=class_sample)

                    if args.apply_scale_snst:
                        if args.scale_type_snst == 'neural_pred':
                            outputs = scale*outputs
                        elif args.scale_type_snst == 'static_mean':
                            static = torch.mean(torch.mean(targets, dim=2, keepdim=True), dim=3, keepdim=True)
                            outputs = 2*static*outputs
                    
                    if args.eval_kl_snst:    
                        loss = F.mse_loss(outputs, targets) + args.kl_ratio_snst * kl_divergence_loss(outputs, targets)
                    else:
                        loss = F.mse_loss(outputs, targets)

                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            if accelerator.is_main_process:
                if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
                    evaluate(args, epoch, accelerator.unwrap_model(model), wz_cls, device=accelerator.device)
                
                if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    os.makedirs(os.path.join(args.output_dir, 'model'), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'ge_model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        },
                        os.path.join(args.output_dir, 'model', 'ckpt.pth'.format(epoch)))

    # start the train loop
    train_loop(args, model, teacher_pipeline, optimizer, dataloader, lr_scheduler, wz_cls)
    # start the train loop with notebook launcher (only when you using notebook to launch the training)
    # config = (args, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    # notebook_launcher(train_loop, config, num_processes=args.num_gpu)

def train_multicolor(args, dataloader, wz_cls):
    model = Multicolor_Optical_Generative_Model(
        img_size=args.sample_size,
        in_channel=args.in_channels,
        num_classes=args.num_classes,
        dim_expand_ratio=128,
        c=args.c, num_masks=args.num_layer,
        wlength_vc=args.wavelength_mtcl,
        ridx_air=args.ridx_air, ridx_mask=args.ridx_layer_mtcl,
        attenu_factor=args.attenu_factor_mtcl,
        total_x_num=args.total_num, total_y_num=args.total_num,
        mask_x_num=args.layer_neuron_num, mask_y_num=args.layer_neuron_num,
        mask_init_method=args.layer_init_method, mask_base_thick=1.0e-3,
        dx=args.dxdy, dy=args.dxdy, 
        object_mask_dist=args.object_layer_dist, 
        mask_mask_dist=args.layer_layer_dist,
        mask_sensor_dist=args.layer_sensor_dist, 
        obj_x_num=args.obj_num, obj_y_num=args.obj_num
    )

    # load the pretrained teacher diffusion
    with open(os.path.join(args.teacher_ckpt_mtcl, 'unet', 'config.json'), 'r') as f:
        unet_config = json.load(f)

    teacher_unet = UNet2DModel(**unet_config)
    teacher_unet_state_dict = load_file(
        os.path.join(args.teacher_ckpt_mtcl, 'unet', 'diffusion_pytorch_model.safetensors'))
    teacher_unet.load_state_dict(state_dict=teacher_unet_state_dict, strict=True)

    with open(os.path.join(args.teacher_ckpt_mtcl, 'scheduler', 'scheduler_config.json'), 'r') as f:
        scheduler_config = json.load(f)

    if args.inference_acc_mtcl:
        noise_scheduler = DDIMScheduler(num_train_timesteps=scheduler_config['num_train_timesteps'],
                                        beta_start=scheduler_config['beta_start'],
                                        beta_end=scheduler_config['beta_end'],
                                        prediction_type=scheduler_config['prediction_type'],
                                        beta_schedule=scheduler_config['beta_schedule'])
    else:
        noise_scheduler = DDPMScheduler(**scheduler_config)

    if wz_cls:
        teacher_pipeline = DDPMPipeline_Costum_ClsEmb(unet=teacher_unet, scheduler=noise_scheduler)
    else:
        teacher_pipeline = DDPMPipeline_Costum(unet=teacher_unet, scheduler=noise_scheduler)

    optimizer = torch.optim.AdamW([{'params': model.DD.parameters(), 'lr': args.learning_rate_optical},
                                   {'params': model.DE.parameters(), 'lr': args.learning_rate_digital}], betas=(0.5, 0.999))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs)
    )

    def evaluate(args, epoch, model, wz_cls, device):
        if wz_cls:
            class_sample = torch.randint(0, args.num_classes, (args.eval_batch_size,), 
                                         generator=torch.manual_seed(args.seed), dtype=torch.int64).to(device)
        else:
            class_sample = None

        noise_sample = torch.randn(args.eval_batch_size, args.in_channels, args.sample_size, args.sample_size, 
                                   generator=torch.manual_seed(args.seed)).to(device=device)

        images, _ = model(noise_sample, class_sample)
        images = images.clip(0, 1)

        img_eval = (images * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
        if args.in_channels == 1:
            pil_img_eval = [Image.fromarray(image.squeeze(), mode='L') for image in img_eval]
        else:
            pil_img_eval = [Image.fromarray(image) for image in img_eval]

        image_grid = make_image_grid(pil_img_eval, 
                                     rows=int(math.sqrt(args.eval_batch_size)), 
                                     cols=int(math.sqrt(args.eval_batch_size)))

        test_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train_loop(args, model, pipeline, optimizer, train_dataloader, lr_scheduler, wz_cls):
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_snapshot")

        model, pipeline.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, pipeline.unet, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        ## training
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                if wz_cls:
                    labels = batch['labels']
                else:
                    labels = None

                with torch.no_grad():
                    if wz_cls:
                        targets, noises, class_sample = \
                            pipeline(batch_size=args.train_batch_size, 
                                    class_sample=labels,
                                    num_inference_steps=int(args.ddpm_num_steps // args.acc_ratio_mtcl), 
                                    output_type='tensor', return_dict=False)
                    else:
                        targets, noises = \
                            pipeline(batch_size=args.train_batch_size, 
                                    num_inference_steps=int(args.ddpm_num_steps // args.acc_ratio_mtcl), 
                                    output_type='tensor', return_dict=False)
                        class_sample = None
                    
                    targets = (targets / 2. + 0.5).clamp(0, 1)
                    noises = noises + args.noise_perturb_mtcl * torch.randn_like(noises) # following stable diffusion, accelerating converging

                with accelerator.accumulate(model):
                    if labels is None:
                        outputs, scale = model(noises)
                    else:
                        outputs, scale = model(noises, labels=class_sample)

                    if args.apply_scale_mtcl:
                        if args.scale_type_mtcl == 'neural_pred':
                            outputs = scale*outputs
                        elif args.scale_type_mtcl == 'static_mean':
                            static = torch.mean(torch.mean(targets, dim=2, keepdim=True), dim=3, keepdim=True)
                            outputs = 2*static*outputs
                    
                    if args.eval_kl_mtcl:    
                        loss = F.mse_loss(outputs, targets) + args.kl_ratio_mtcl * kl_divergence_loss(outputs, targets)
                    else:
                        loss = F.mse_loss(outputs, targets)

                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            if accelerator.is_main_process:
                if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
                    evaluate(args, epoch, accelerator.unwrap_model(model), wz_cls, device=accelerator.device)
                
                if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    os.makedirs(os.path.join(args.output_dir, 'model'), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'ge_model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        },
                        os.path.join(args.output_dir, 'model', 'ckpt.pth'.format(epoch)))

    # start the train loop
    train_loop(args, model, teacher_pipeline, optimizer, dataloader, lr_scheduler, wz_cls)
    # start the train loop with notebook launcher (only when you using notebook to launch the training)
    # config = (args, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    # notebook_launcher(train_loop, config, num_processes=args.num_gpu)

def train_iterative(args, dataloader, wz_cls):
    model = Iterative_Optical_Generative_Model(
        img_size=args.sample_size,
        in_channel=args.in_channels,
        num_classes=args.num_classes,
        dim_expand_ratio=128,
        c=args.c, num_masks=args.num_layer,
        wlength_vc=args.wavelength_mtcl,
        ridx_air=args.ridx_air, ridx_mask=args.ridx_layer_mtcl,
        attenu_factor=args.attenu_factor_mtcl,
        total_x_num=args.total_num, total_y_num=args.total_num,
        mask_x_num=args.layer_neuron_num, mask_y_num=args.layer_neuron_num,
        mask_init_method=args.layer_init_method, mask_base_thick=1.0e-3,
        dx=args.dxdy, dy=args.dxdy, 
        object_mask_dist=args.object_layer_dist, 
        mask_mask_dist=args.layer_layer_dist,
        mask_sensor_dist=args.layer_sensor_dist, 
        obj_x_num=args.obj_num, obj_y_num=args.obj_num,
        time_embedding_type=args.time_embedding_type_o,
        num_train_timesteps=args.ddpm_num_steps
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, 
                                    beta_schedule=args.ddpm_beta_schedule,
                                    prediction_type=args.prediction_type_o,
                                    beta_start=args.beta_start_itrt,
                                    beta_end=args.beta_end_itrt)
    
    optimizer = torch.optim.AdamW([{'params': model.DD.parameters(), 'lr': args.learning_rate_optical},
                                   {'params': model.DE.parameters(), 'lr': args.learning_rate_digital}], betas=(0.5, 0.999))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs)
    )

    def evaluate(args, epoch, pipeline):
        images = pipeline(
            batch_size=args.eval_batch_size,
            generator=torch.manual_seed(args.seed),
            num_inference_steps=args.ddpm_num_steps
        ).images

        image_grid = make_image_grid(images, 
                                     rows=int(math.sqrt(args.eval_batch_size)), 
                                     cols=int(math.sqrt(args.eval_batch_size)))

        test_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, wz_cls):
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_iterative")

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        ## training
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                if wz_cls:
                    labels = batch['labels']
                else:
                    labels = None
                
                noise = torch.randn_like(clean_images)
    
                batch_size = clean_images.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                    device=clean_images.device, dtype=torch.int64)
                
                # add noise according to the noise magnitude of each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):

                    noise_pred = model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]

                    if args.prediction_type_o == "epsilon":    
                        loss = F.mse_loss(noise_pred, noise)
                    elif args.prediction_type_o == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(noise_pred.float(), 
                                                        clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            if accelerator.is_main_process:
                if wz_cls:
                    pipeline = DDPMPipeline_Costum_ClsEmb(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                else:
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
                    evaluate(args, epoch, pipeline)
                
                if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    pipeline.save_pretrained(args.output_dir)

    # start the train loop
    train_loop(args, model, noise_scheduler, optimizer, dataloader, lr_scheduler, wz_cls)
    # start the train loop with notebook launcher (only when you using notebook to launch the training)
    # config = (args, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    # notebook_launcher(train_loop, config, num_processes=args.num_gpu)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)