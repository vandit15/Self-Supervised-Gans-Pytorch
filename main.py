"""
    Pytorch implementation of Self-Supervised GAN
    Reference: "Self-Supervised GANs via Auxiliary Rotation Loss"
    Authors: Ting Chen,
                Xiaohua Zhai,
                Marvin Ritter,
                Mario Lucic and
                Neil Houlsby
    https://arxiv.org/abs/1811.11212 CVPR 2019.
    Script Author: Vandit Jain. Github:vandit15
"""
import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from model import Generator, Discriminator
from training import Trainer

data_loader, _ = get_mnist_dataloaders(batch_size=2)
img_size = (32, 32, 1)

generator = Generator(resnet = True, z_size = 128, channel = 1)
discriminator = Discriminator(resnet = True, spectral_normed = True, num_rotation = 4,
				 channel = 1, ssup = True)


# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  weight_rotation_loss_d = 1.0, weight_rotation_loss_g = 0.5,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=False)

# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')