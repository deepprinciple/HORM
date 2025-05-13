from uuid import uuid4
from copy import deepcopy
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from training_module import PotentialModule


torch.set_float32_matmul_precision('high')
model_type = "AlphaNet"
version = "jac2"
project = "horm-test"
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

model_config = dict(
    name=model_type,
    num_targets=1,
    output_dim=1,
    readout="sum",
    use_pbc=False,
    direct_forces=False,
    eps=1e-6,
    num_layers=4,
    hidden_channels=128,
    cutoff=5.0,
    num_radial=64,
    use_sigmoid=False,
    head=16,
    a=0.35,  # forces.std()
    pos_require_grad=True,
    b=0,  # energy.mean()
    main_chi1=32,
    mp_chi1=32,
    chi2=8,
    hidden_channels_chi=96,
    has_dropout_flag=True,  # 重要！推理时关闭 dropout
    has_norm_before_flag=True,
    has_norm_after_flag=False,
    reduce_mode="sum",
    device="cuda",
    compute_forces=True,
    compute_stress=False,
)

optimizer_config = dict(
    lr=5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

training_config = dict(
    trn_path="data/sample_100.lmdb",
    val_path="data/sample_100.lmdb",
    bz=2,
    num_workers=48,
    clip_grad=True,
    gradient_clip_val=0.1,
    ema=False,
    lr_schedule_type="step",
    lr_schedule_config=dict(
        gamma=0.85,
        step_size=50,
    ),
)

pm = PotentialModule(model_config, optimizer_config, training_config)



logger = wandb_logger = WandbLogger(
    project=project,
    log_model=False,
    name=run_name,

)

ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}"

checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ff-{epoch:03d}-{val-totloss:.4f}-{val-MAE_E:.4f}-{val-MAE_F:.4f}",
    every_n_epochs=10,
    save_top_k=-1,
)

early_stopping_callback = EarlyStopping(
    monitor="val-totloss",
    patience=1000,
    mode="min",
)

lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [
    checkpoint_callback,
    early_stopping_callback,
    TQDMProgressBar(),
    lr_monitor,
]

strategy = "ddp_find_unused_parameters_true"
trainer = Trainer(
    devices=1,
    num_nodes=1,
    accelerator="gpu",
    strategy=strategy,
    max_epochs=10000,
    callbacks=callbacks,
    default_root_dir=ckpt_path,
    logger=logger,
    gradient_clip_val=0.1,
    accumulate_grad_batches=1,
    limit_train_batches=1600,
    limit_val_batches=80,
)

# ckpt_path = f"/home/ubuntu/efs/deep-principle/pkgs/MLFF/AlphaNet/checkpoint/TS1x-FF-debugging/AlphaNet-4L40-step-6208f886ecfb/ff-epoch=589-val-totloss=5.2281-val-MAE_E=0.0305-val-MAE_F=0.0511.ckpt"
# trainer.fit(pm, ckpt_path=ckpt_path)
trainer.fit(pm)