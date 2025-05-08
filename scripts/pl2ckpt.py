import torch

model = torch.load('epoch=175-val_loss=22.3781-val_energy_mae=1.5878-val_force_mae=0.1285.ckpt')
state_dict = model['state_dict']
print(state_dict.keys())
new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
torch.save(new_state_dict, 't5.ckpt')

print("State_dict keys updated and saved successfully.")

