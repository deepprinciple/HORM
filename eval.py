from training_module import PotentialModule, compute_extra_props
from torch_geometric.loader import DataLoader
from ff_lmdb import LmdbDataset
import torch
from tqdm import tqdm


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad([y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph)[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    """Compute Hessian matrix using autograd."""
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)
    
    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]
    
    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)
        
    # Stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def hess2eigenvalues(hess):
    """Convert Hessian to eigenvalues with proper unit conversion"""
    hartree_to_ev = 27.2114
    bohr_to_angstrom = 0.529177
    ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev
    
    hess = hess * ev_angstrom_2_to_hartree_bohr_2
    eigen_values, _ = torch.linalg.eigh(hess)
    return eigen_values


def evaluate(lmdb_path,  checkpoint_path):

    ckpt = torch.load(checkpoint_path)
    model_name =ckpt['hyper_parameters']['model_config']['name']

    print(f"Model name: {model_name}")

    pm = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to('cuda')

    dataset = LmdbDataset(lmdb_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize metrics
    total_e_error = 0.0
    total_f_error = 0.0
    total_h_error = 0.0
    total_eigen_error = 0.0
    total_asymmetry_error = 0.0
    n_samples = 0

    idx = 0

    for batch in tqdm(dataloader, desc='Evaluating', total=len(dataloader)):

        if idx >= 200:
            break
        else:
            idx += 1

        batch = batch.to('cuda')
        batch.pos.requires_grad_()
        batch = compute_extra_props(batch)

        # Forward pass
        if model_name == 'LEFTNet':
            ener, force = pm.forward_autograd(batch)
        else:
            ener, force = pm.forward(batch)
        
        # Compute hessian and eigenvalues
        # Use reshape instead of view to handle non-contiguous tensors
        hess = compute_hessian(batch.pos, ener, force)
        eigenvalues = hess2eigenvalues(hess)

        # Compute errors
        e_error = torch.mean(torch.abs(ener.squeeze() - batch.ae))
        f_error = torch.mean(torch.abs(force - batch.forces))
        
        # Reshape true hessian
        n_atoms = batch.pos.shape[0]
        hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
        h_error = torch.mean(torch.abs(hess - hessian_true))
        
        # Eigenvalue error
        eigen_true = hess2eigenvalues(hessian_true)
        eigen_error = torch.mean(torch.abs(eigenvalues - eigen_true))

        # Asymmetry error
        asymmetry_error = torch.mean(torch.abs(hess - hess.T))
        total_asymmetry_error += asymmetry_error.item()

        # Update totals
        total_e_error += e_error.item()
        total_f_error += f_error.item()
        total_h_error += h_error.item()
        total_eigen_error += eigen_error.item()
        n_samples += 1

        # Memory management
        torch.cuda.empty_cache()

    # Calculate average errors
    mae_e = total_e_error / n_samples
    mae_f = total_f_error / n_samples
    mae_h = total_h_error / n_samples
    mae_eigen = total_eigen_error / n_samples
    mae_asymmetry = total_asymmetry_error / n_samples

    print(f"\nResults:")
    print(f"Energy MAE: {mae_e:.6f}")
    print(f"Forces MAE: {mae_f:.6f}")
    print(f"Hessian MAE: {mae_h:.6f}")
    print(f"Eigenvalue MAE: {mae_eigen:.6f}")
    print(f"Asymmetry MAE: {mae_asymmetry:.6f}")


if __name__ == '__main__':
    # checkpoint_path = 'eqv2_efh.ckpt'
    torch.manual_seed(42)

    checkpoint_path = 'ckpt/left.ckpt'
    lmdb_path = 'data/sample_100.lmdb'
    
    evaluate(lmdb_path, checkpoint_path)