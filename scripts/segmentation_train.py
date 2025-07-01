import sys
import argparse
import yaml
from argparse import Namespace
from accelerate import Accelerator

sys.path.append("../")
sys.path.append("./")

import torch as th
import torchvision.transforms as transforms
import wandb
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from guided_diffusion.train_util import TrainLoop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to training config YAML')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    defaults = create_defaults()
    defaults.update(cfg)
    args = Namespace(**defaults)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=defaults)

    accelerator = Accelerator(
        mixed_precision="fp16" if args.use_fp16 else "no"
    )

    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_train = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size))]
        transform_train = transforms.Compose(tran_list)
        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory:", args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    if len(ds) == 0:
        raise ValueError(f"No data found in directory: {args.data_dir}")

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    data = iter(datal)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(accelerator.device)
    wandb.watch(model, log="all")

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=args.diffusion_steps
    )

    accelerator.print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_wandb=True,
    ).run_loop()

def create_defaults():
    defaults = dict(
        data_name='BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu="0,1",
        out_dir='./results/',
        wandb_project='MedSeg',
        wandb_run_name='training'
    )
    defaults.update(model_and_diffusion_defaults())
    return defaults

if __name__ == "__main__":
    main()
