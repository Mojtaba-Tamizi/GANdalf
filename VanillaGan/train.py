import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
from torch.autograd.variable import Variable

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCHSIZE = 128
EPOCHS = 300
LATENT_DIM = 100  # Dimension of the noise vector
LR = 0.0003  # Learning rate for Adam optimizer
BETA1 = 0.5  # Beta1 hyperparameter for Adam optimizer

# Data transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

# Load MNIST dataset
dataset = datasets.MNIST(root="../Datasets/MNIST", train=True, transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)

# Initialize generator, discriminator, and TensorBoard writer
generator = Generator(LATENT_DIM).to(DEVICE)
discriminator = Discriminator().to(DEVICE)
writer = SummaryWriter("runs/vanilla_gan")

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

# Fixed noise for generating images to track progression over epochs
fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

# Function to generate random noise
def generate_noise(batch_size, latent_dim):
    return Variable(torch.randn(batch_size, latent_dim)).to(DEVICE)

# Training Loop
for epoch in range(EPOCHS):
    d_loss_list, g_loss_list = [], []
    for batch_idx, (real_images, _) in enumerate(loader):
        real_images = real_images.to(DEVICE)  # Move real images to the device
        batch_size = real_images.size(0)

        # Flatten the real images to (batch_size, 784)
        real_images_flat = real_images.view(batch_size, -1)

        # Generate noise and create fake images
        noise = generate_noise(batch_size, LATENT_DIM).to(DEVICE)
        fake_images = generator(noise).view(batch_size, -1)  # Ensure shape is (batch_size, 784)

        # Train Discriminator on Real Images
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=DEVICE)  # Real labels
        outputs_real = discriminator(real_images_flat)  # Discriminator output for real images
        d_loss_real = criterion(outputs_real, real_labels)  # Loss for real images
        d_loss_real.backward()  # Retain graph for the next backward pass

        # Train Discriminator on Fake Images
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)  # Fake labels
        outputs_fake = discriminator(fake_images.detach())  # Discriminator output for fake images
        d_loss_fake = criterion(outputs_fake, fake_labels)  # Loss for fake images
        d_loss_fake.backward()  # Backpropagate the loss for fake images
        
        # Update Discriminator
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)  # Discriminator output for fake images (without detach)
        g_loss = criterion(outputs, real_labels)  # We want the generator to fool the discriminator
        g_loss.backward()  # Backpropagate the generator loss
        optimizer_G.step()

        # Log losses to TensorBoard
        d_loss = d_loss_real + d_loss_fake  # Total discriminator loss
        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())

        # Print training stats every few batches
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}], Batch [{batch_idx}/{len(loader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    writer.add_scalar("Loss/Discriminator", sum(d_loss_list)/len(d_loss_list), epoch)
    writer.add_scalar("Loss/Generator", sum(g_loss_list)/len(g_loss_list), epoch)
    # Generate and save images to TensorBoard at the end of each epoch
    with torch.no_grad():
        fake_images_fixed = generator(fixed_noise).view(-1, 1, 28, 28)  # Generate images with fixed noise
        img_grid = utils.make_grid(fake_images_fixed, normalize=True, value_range=(-1, 1))
        writer.add_image("Generated Images", img_grid, global_step=epoch)

# Close TensorBoard writer
writer.close()

torch.save(generator.state_dict(), f"generator_epoch_{EPOCHS}.pth")
torch.save(discriminator.state_dict(), f"discriminator_epoch_{EPOCHS}.pth")
