import torch
from data.garment_dataset import GarmentDataset
from data.structures import GarmentDict
from torch.utils.data import DataLoader, random_split
import json
import torch.nn.functional as F
from model.VDM_latent import VDMStableDiffusionPipeline
from data.normalization import flatten_embeddings
import os
import argparse
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from utils.uv_tools import apply_displacement
from utils.geometry import l2dist
import wandb

def main(config_file=None):

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    description = 'Training script for D-Garment'
    parser = argparse.ArgumentParser(description=description,
                                     prog='D-Garment training')
    parser.add_argument('config', type=str,
                        help='The config file for training')

    args = parser.parse_args()

    config_file = args.config

    with open(config_file) as f:
        config = json.load(f)

    resume_epoch = os.path.exists(os.path.join(config["output"]["folder"], "checkpoint.pt"))

    dataset = GarmentDataset(config, device="cpu")
    train_dataset, eval_dataset = random_split(dataset, [1. - config["validation"]["proportion"],
                                                         config["validation"]["proportion"]],
                                                         generator=torch.Generator('cpu').manual_seed(0))

    print("Using train/eval split: " + str(len(train_dataset)) + " / " + str(len(eval_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], drop_last=True, num_workers=4, prefetch_factor=2, shuffle=True, persistent_workers=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["validation"]["batch_size"], drop_last=True, num_workers=2, prefetch_factor=2, shuffle=True, persistent_workers=True, pin_memory=True)

    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_with="wandb",
        project_dir=os.path.join(config["output"]["folder"], "logs"),
    )

    if not resume_epoch:
        # define and create initial models, reset model parameters!!!
        # static_dimension = 16 + 3 # beta (body shape) + cloth material (bending, stretching, density)
        # dynamic_dimension = 3 + 22 * 6 # trans + theta (body pose)
        # condition_dimension = static_dimension + dynamic_dimension # = 151

        # input_encoder = InputAdapter(config["input_encoder"]['static_dim'], config["input_encoder"]['dynamic_dim'], config["model"]['cross_attention_dim'], config["input_encoder"]['hidden_dim'])

        unet = UNet2DConditionModel(
            sample_size=config["model"]["image_size"]//config["vae"]["autoencoder_scale"],    # image_size should be multiple of 8
            in_channels=4,
            out_channels=4,
            # encoder_hid_dim=config["model"]["encoder_hid_dim"],
            cross_attention_dim=config["model"]["cross_attention_dim"],
            attention_head_dim=config["model"]["attention_head_dim"],
            block_out_channels=(64, 128, 256, 512, 512), # the number of output channels for each Conv2D block
                            # 128, 128, 256, 256, 512, 512 in DiffusedWrinkles
                            # 320, 640, 1280, 1280 in Realistic_Vision_V4.0
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",  # a ResNet downsampling block with cross-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            )
        )

        vae = AutoencoderKL.from_pretrained(config["vae"]["folder"])

        # num_train_timesteps is commonly 1000, use 100 for faster training
        scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2', clip_sample=False, num_train_timesteps=config["training"]["num_train_timesteps"])

        stable_diff = VDMStableDiffusionPipeline(vae, unet, scheduler)

        if accelerator.is_main_process:
            out_dir = config["output"]["folder"]
            os.makedirs(out_dir, exist_ok=True)
            copy_file = os.path.join(out_dir, os.path.basename(config_file))
            with open(copy_file, "w") as f:
                json.dump(config, f, indent=4)

            stable_diff.save_pretrained(out_dir)

            print("Unet network size: " + str(stable_diff.unet.num_parameters()/1e6) + " million")
            print("VAE network size: " + str(stable_diff.vae.num_parameters()/1e6) + " million")

    accelerator.wait_for_everyone()
    stable_diff = VDMStableDiffusionPipeline.from_pretrained(config["output"]["folder"])
    
    vae = stable_diff.vae
    unet = stable_diff.unet
    noise_scheduler = stable_diff.scheduler

    optimizer = torch.optim.AdamW(unet.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = config["training"]["lr_warmup_steps"],
        num_training_steps = len(train_dataloader) * config["training"]["max_epochs"]
    )  

    resumed_epoch = 0
    if resume_epoch:
        checkpoint = torch.load(os.path.join(config["output"]["folder"], 'checkpoint.pt'), weights_only=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resumed_epoch = checkpoint['epoch']

        print("Resuming epoch: " + str(resumed_epoch))

    torch.random.manual_seed(0)
    
    vae.requires_grad_(False)
    unet.requires_grad_(True)
    
    vae.eval()
    unet.train()

    vae.to(accelerator.device, memory_format=torch.channels_last)
    vae.decode = torch.compile(vae.decode, mode="max-autotune", fullgraph=True, dynamic=True)
    vae.encode = torch.compile(vae.encode, mode="max-autotune", fullgraph=True, dynamic=True)
    # wrappedModels.unet = torch.compile(wrappedModels.unet, mode="reduce-overhead", dynamic=True)
    # train_dataset = TrainingDataset(train_dataset, vae)


    if accelerator.is_main_process:
        os.makedirs(config["output"]["folder"], exist_ok=True)
        # os.makedirs(os.path.join(config["output"]["folder"], "samples"), exist_ok=True)
        accelerator.init_trackers("garment",
            config={
            "scheduler": noise_scheduler.config,
            "config": config
            },
            init_kwargs={"wandb": {"id": os.path.basename(config["output"]["folder"])}}
        )
        # wandb.watch(unet, log='all', log_freq=1000, log_graph=False)

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(resumed_epoch, config["training"]["max_epochs"]):
        progress_bar.reset()
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            batch:GarmentDict
            optimizer.zero_grad(set_to_none=True)
            # latents = batch["latents"].to(accelerator.device, non_blocking=True)
            
            with torch.no_grad():

                latents = vae.encode(batch.vdm.permute(0,3,1,2)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor   # https://github.com/huggingface/diffusers/issues/437

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(unet):

                # Predict the noise residual
                states = flatten_embeddings(batch, material=config['model']['use_material'])
                pred = unet(noisy_latents, timesteps, encoder_hidden_states=states, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                
                loss = F.mse_loss(pred.float(), target.float(), reduction='mean') # mse_loss or l1_loss or huber_loss or smooth_l1_loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
            
            if accelerator.is_main_process:
                progress_bar.update(1)
                logs = {"loss": loss.detach().item() , "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                global_step += 1

            if (step % config["training"]["save_freq"] == 0 or step == len(train_dataloader) - 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.unwrap_model(unet).save_pretrained(os.path.join(config["output"]["folder"], "unet"))
                    checkpoint = { 
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()}
                    torch.save(checkpoint, os.path.join(config["output"]["folder"], 'checkpoint.pt'))
        
            if (step % config["validation"]["eval_freq"] == 0 or step == len(train_dataloader) - 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # eval
                    with torch.inference_mode():
                        pipeline = VDMStableDiffusionPipeline(unet=accelerator.unwrap_model(unet),
                                                            vae=accelerator.unwrap_model(vae),
                                                            scheduler=noise_scheduler,)
                        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")
                        pipeline.set_progress_bar_config(disable=True)

                        eval_data = next(iter(eval_dataloader))

                        img = pipeline(eval_data, num_inference_steps=20, generator=torch.Generator(device='cpu').manual_seed(0), use_material=config['model']['use_material'])

                        examples = []
                        image = wandb.Image(img[0].permute(2,0,1), caption="Generated")
                        examples.append(image)
                        image = wandb.Image(eval_data.vdm[0].permute(2,0,1), caption="Ground truth")
                        examples.append(image)

                        estimated_vertices = apply_displacement(dataset.template, img, dataset.img_size).cpu()
                        ground_truth = eval_data.cloth_vertices.cpu()
                        eval_loss = l2dist(estimated_vertices, ground_truth)

                        # dist = eval_loss[0].clone()
                        # dist = dist - dist.min()
                        # dist = dist / dist.max()
                        # dist = dist.cpu() * 3
                        # rgb = plt.cm.jet(dist)[:,:-1]*255

                        # estimated_vertices_colored = np.concat((estimated_vertices[0].cpu(), rgb), axis=-1)
                        # wandb.log({"point_cloud": wandb.Object3D(estimated_vertices_colored)})
                        eval_loss = eval_loss.mean()

                    logs = {"eval_dist": eval_loss.detach().item(), "Images": examples}
                    accelerator.log(logs, step=global_step)

                    # torch.save(img.cpu(), os.path.join(config["output"]["folder"], 'samples/step_' + str(step) + '.pt'))

                    del pipeline
                    torch.cuda.empty_cache()

if __name__ == "__main__": 
    main()