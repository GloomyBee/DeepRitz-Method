from core.loss.losses import MonteCarloEnergyLoss
import torch

loss_compute =MonteCarloEnergyLoss()
a =[2,1]
b =[2,2]
c =[1,1]

output =loss_compute.compute_energy_loss(torch.a,torch.b)

print(output)

