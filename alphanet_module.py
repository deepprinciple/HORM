'''
PyTorch Lightning module for training AlphaNet
'''
from typing import Dict, List, Optional, Tuple
import pdb
from pathlib import Path
import torch
from torch import nn

from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, CosineAnnealingLR
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, CosineSimilarity
from torch_scatter import scatter_mean

from ff_lmdb import LmdbDataset
from utils import average_over_batch_metrics, pretty_print
import utils as diff_utils
from alphanet.models.alphanet import AlphaNet


LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}
GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])


def compute_extra_props(batch, pos_require_grad=True):
    device = batch.energy.device
    indices = batch.one_hot.long().argmax(dim=1)
    batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
    batch.pos = remove_mean_batch(batch.pos, batch.batch)
    if pos_require_grad:
        batch.pos.requires_grad_(True)
    return batch


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


class AlphaConfig:
    def __init__(self, config):
        for k,v in config.items():
            setattr(self, k, v)


class PotentialModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        super().__init__()


        self.model_config = model_config

        if self.model_config['name'] == 'EquiformerV2':
            import yaml
            with open('./equiformer_v2.yml', 'r') as file:
                config = yaml.safe_load(file)
            model_config = config['model']
            self.potential = EquiformerV2_OC20(**model_config) 
        elif self.model_config['name'] == 'Alphanet':
            self.potential = AlphaNet(
                AlphaConfig(model_config)
            ).float()
        elif self.model_config['name'] =='LeftNet':
            from leftnet.potential import Potential
            self.potential = Potential(
                model_config=model_config,
                node_nfs=model_config['node_nfs'],
                edge_nf=model_config['edge_nf'],
                condition_nf=model_config['condition_nf'],
                fragment_names=model_config['fragment_names'],
                pos_dim=model_config['pos_dim'],
                edge_cutoff=model_config['edge_cutoff'],
                model=model_config['model'],
                enforce_same_encoding=model_config['enforce_same_encoding'],
                source=model_config['source'],
                timesteps=model_config['timesteps'],
                condition_time=model_config['condition_time'],
            )
        else:
            print("Please Check your model name (choose from 'EquiformerV2', 'Alphanet', 'LeftNet')")           
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.pos_require_grad = model_config["pos_require_grad"]

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = diff_utils.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

        self.loss_fn = nn.L1Loss()
        self.MAEEval = MeanAbsoluteError()
        self.MAPEEval = MeanAbsolutePercentageError()
        self.cosineEval = CosineSimilarity(reduction="mean")
        self.val_step_outputs = []
        from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.potential.parameters(),
            **self.optimizer_config
        )

        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer,
                **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ts1x_hess_train_big.lmdb"),
                **self.training_config,
            )
            self.val_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_valid_5percent_with_hessian.lmdb"),
                **self.training_config,
            )
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))
        elif stage == "test":
            self.test_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_test.lmdb"),
                **self.training_config,
            )
        else:
            raise NotImplementedError
    def get_jacobian(self,forces, pos, grad_outputs, create_graph=False, looped=False):
        # This function should: take the derivatives of forces with respect to positions. 
        # Grad_outputs should be supplied. if it's none, then
        def compute_grad(grad_output):
            return torch.autograd.grad(
                    outputs=forces,
                    inputs=pos,
                    grad_outputs=grad_output,
                    create_graph=create_graph,
                    retain_graph=True
                )[0]
        if not looped:
            if len(grad_outputs.shape) == 4:
                compute_jacobian = torch.vmap(torch.vmap(compute_grad))
            else:
                compute_jacobian = torch.vmap(compute_grad)
            return compute_jacobian(grad_outputs)
        else:
            num_atoms = forces.shape[0]
            if len(grad_outputs.shape) == 4:
                full_jac = torch.zeros(grad_outputs.shape[0], 3, num_atoms, 3).to(forces.device)
                for i in range(grad_outputs.shape[0]):
                    for j in range(3):
                        full_jac[i, j] = compute_grad(grad_outputs[i, j])
            else:
                full_jac = torch.zeros(grad_outputs.shape[0], num_atoms, 3).to(forces.device)
                for i in range(grad_outputs.shape[0]):
                        full_jac[i] = compute_grad(grad_outputs[i])
            return full_jac
    def get_force_jac_loss(self,forces, batch, hessian_label, num_samples=2, looped=False, finite_differences=False, forward=None, collater=None):

        natoms = batch.natoms
        total_num_atoms = forces.shape[0]

        mask = torch.ones(total_num_atoms, dtype=torch.bool)
        cumulative_sums = [0] + torch.cumsum(natoms, 0).tolist()
        
        by_molecule = []
        grad_outputs = torch.zeros((num_samples, total_num_atoms, 3)).to(forces.device)
        for i, atoms_in_mol in enumerate(batch.natoms):
            submask = mask[cumulative_sums[i]:cumulative_sums[i+1]]
            samples = self.sample_with_mask(atoms_in_mol, num_samples, submask)
            
            by_molecule.append(samples) # swap below and above line, crucial
            offset_samples = samples.clone()  # Create a copy of the samples array to avoid modifying the original
            offset_samples[:, 0] += cumulative_sums[i]
            # Vectorized assignment to grad_outputs
            grad_outputs[torch.arange(samples.shape[0]), offset_samples[:, 0], offset_samples[:, 1]] = 1
        # Compute the jacobian using grad_outputs
        
        jac = self.get_jacobian(forces, batch.pos, grad_outputs, create_graph=True, looped=looped)
        #jac = self.get_jacobian_finite_difference(forces, batch, grad_outputs = grad_outputs, forward=self._forward)


        # Decomposing the Jacobian tensor by molecule in a batch
        mask_per_mol = [mask[cum_sum:cum_sum + nat] for cum_sum, nat in zip(cumulative_sums[:-1], natoms)]
        num_free_atoms_per_mol = torch.tensor([sum(sub_mask) for sub_mask in mask_per_mol], device=natoms.device)
        cum_jac_indexes = [0] +  torch.cumsum((num_free_atoms_per_mol * natoms)*9, dim=0).tolist()
        
        jacs_per_mol = [jac[:len(mol_samps), cum_sum:cum_sum + nat, :] for mol_samps, cum_sum, nat in zip(by_molecule, cumulative_sums[:-1], natoms)]
        jacs_per_mol = [mol_jac[:, mask, :] for mol_jac, mask in  zip(jacs_per_mol, mask_per_mol)] # do the same for te student hessians

        if torch.any(torch.isnan(jac)):
            raise Exception("FORCE JAC IS NAN")
        
        batch.fixed = torch.zeros(total_num_atoms)

        true_jacs_per_mol = []
        for i, samples in enumerate(by_molecule):
            fixed_atoms = batch.fixed[cumulative_sums[i]:cumulative_sums[i+1]]
            fixed_cumsum = torch.cumsum(fixed_atoms, dim=0)
            num_free_atoms = num_free_atoms_per_mol[i]
            curr = hessian_label[cum_jac_indexes[i]:cum_jac_indexes[i+1]].reshape(num_free_atoms, 3, natoms[i], 3)
            curr = curr[:, :, mask_per_mol[i], :] # filter out the masked columns 
            subsampled_curr = curr[(samples[:, 0] - fixed_cumsum[samples[:, 0]]).long(), samples[:, 1]] # get the sampled rows
            true_jacs_per_mol.append(subsampled_curr)

        # just copying what DDPLoss does for our special case
        custom_loss = lambda jac, true_jac: torch.norm(jac - true_jac, p=2, dim=-1).sum(dim=1).mean(dim=0)
        losses = [custom_loss(-jac, true_jac) for jac, true_jac in zip(jacs_per_mol, true_jacs_per_mol)]
        valid_losses = [loss * 1e-8 if true_jac.abs().max().item() > 10000 else loss for loss, true_jac in zip(losses, true_jacs_per_mol)]  # filter weird hessians
        
        loss = sum(valid_losses)




        num_samples = (batch.batch.max()+1)
            # Multiply by world size since gradients are averaged
            # across DDP replicas
        loss  = loss / num_samples / 10
        return loss
 
    def sample_with_mask(self, n, num_samples, mask):
        if mask.shape[0] != n:
            raise ValueError("Mask length must be equal to the number of rows in the grid (n)")
        
        # Calculate total available columns after applying the mask
        # Only rows where mask is True are considered
        valid_rows = torch.where(mask)[0]  # Get indices of rows that are True
        if valid_rows.numel() == 0:
            raise ValueError("No valid rows available according to the mask")

        # Each valid row contributes 3 indices
        valid_indices = valid_rows.repeat_interleave(3) * 3 + torch.tensor([0, 1, 2]).repeat(valid_rows.size(0)).to(mask.device)

        # Sample unique indices from the valid indices
        chosen_indices = valid_indices[torch.randperm(valid_indices.size(0))[:num_samples]]

        # Convert flat indices back to row and column indices
        row_indices = chosen_indices // 3
        col_indices = chosen_indices % 3

        # Combine into 2-tuples
        samples = torch.stack((row_indices, col_indices), dim=1)
        
        return samples
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()
        batch = compute_extra_props(
            batch, 
            pos_require_grad=self.pos_require_grad
        )
        hat_ae, hat_forces = self.potential.forward(
            batch.to(self.device),
        )
        hat_ae = hat_ae.squeeze().to(self.device)
        hat_forces = hat_forces.to(self.device)
        ae = batch.ae.to(self.device)
        forces = batch.forces.to(self.device)
        hessian_loss = self.get_force_jac_loss(hat_forces, batch, batch.hessian)
        eloss = self.loss_fn(ae, hat_ae)
        floss = self.loss_fn(forces, hat_forces)
        info = {
            "MAE_E": eloss.detach().item(),
            "MAE_F": floss.detach().item(),
            "MAE_hessian":hessian_loss.detach().item(),
        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()
        
        loss = floss * 100 + eloss * 4 + hessian_loss * 4
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)

        self.log("train-totloss", loss, rank_zero_only=True)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        del info
        return loss

    def __shared_eval(self, batch, batch_idx, prefix, *args):
      with torch.enable_grad():
        loss, info = self.compute_loss(batch)
        info["totloss"] = loss.item()

        info_prefix = {}
        for k, v in info.items():
            key = f"{prefix}-{k}"
            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.is_cuda:
                    v = v.cpu()
                if v.numel() == 1:
                    info_prefix[key] = v.item()
                else:
                    info_prefix[key] = v.numpy()
            else:
                info_prefix[key] = v
            self.log(key, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        del info
      return info_prefix
  
    def _shared_eval(self, batch, batch_idx, prefix, *args):
        loss, info = self.compute_loss(batch)
        detached_loss = loss.detach()
        info["totloss"] = detached_loss.item()
       # info["totloss"] = loss.item()
        
        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        del info

        if torch.cuda.is_available():
           torch.cuda.empty_cache()
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # 收集每个 batch 的输出
        self.val_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        
        val_epoch_metrics = average_over_batch_metrics(self.val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")

        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)
    
        self.val_step_outputs.clear()
        

    def _configure_gradient_clipping(
        self,
        optimizer,
        # optimizer_idx,
        gradient_clip_val,
        gradient_clip_algorithm
    ):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 2 * self.gradnorm_queue.mean() + \
            3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = diff_utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')