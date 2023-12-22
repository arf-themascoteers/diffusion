import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import cif
import view

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = cif.get_planes()
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
print(sampled_images.shape) # (4, 3, 128, 128)
sampled_images = sampled_images.cpu().detach()
# Save the tensor to a file using torch.save
torch.save(sampled_images, 'saved_tensor.pt')

for sample in sampled_images:
    view.view_it(sample)