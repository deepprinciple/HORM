'''
Fine tuning the AlphaNet model on the transition1x dataset
'''

from typing import List, Optional, Tuple
from collections import OrderedDict
from uuid import uuid4
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from alphanet_module import PotentialModule


model_type = "AlphaNet"
version = "4L40"
project = "TS1x-FF-debugging"
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

model_config = dict(
    name = "Alphanet",
    num_targets = 1,
    output_dim = 1,
    readout = "sum",
    use_pbc = False,
    direct_forces= False,
    eps = 1e-6,
    num_layers = 4,
    hidden_channels = 128,
    cutoff = 5.0,
    pos_require_grad = True,
    num_radial = 96,
    use_sigmoid = False,
    head = 16,
    a = 0.35,  # forces.std()
    b = 0,  # energy.mean()
    main_chi1 = 32,
    mp_chi1 = 32,
    chi2 = 8,
    hidden_channels_chi = 96,
    has_dropout_flag = True,  # Important! turn off dropout when inference
    has_norm_before_flag = True,  
    has_norm_after_flag = False,
    reduce_mode = "sum",
    device = "cuda",
    compute_forces = True,
    compute_stress = False,
)

optimizer_config = dict(
    lr=2.5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

training_config = dict(
    datadir="/home/ubuntu/efs/OA_ReactDiff/oa_reactdiff/data/transition1x/ff",
    bz=32,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    lr_schedule_type=None,
    # lr_schedule_config=dict(
    #     gamma=0.8,
    #     step_size=10,
    # ),
    # lr_schedule_config=dict(
    #     T_0 = 100,
    #     eta_min=1e-6,
    #     T_mult=1,
    # )
)

pm = PotentialModule(model_config, optimizer_config, training_config)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
        entity="deep-principle",
    )

    try:
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(
            pm.potential, log="all", log_freq=100, log_graph=False,
        )
    except:
        pass
    

ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}"
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ff-{epoch:03d}-{val-totloss:.4f}-{val-MAE_E:.4f}-{val-MAE_F:.4f}",
    every_n_epochs=100,
    save_top_k=-1,
)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
# if training_config["ema"]:
#     callbacks.append(EMACallback())

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy = None
print(strategy, devices)
trainer = Trainer(
    max_epochs=10000,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=20,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=400,
    limit_val_batches=20,
    # resume_from_checkpoint=checkpoint_path,
    # max_time="00:10:00:00",
)

trainer.fit(pm)
