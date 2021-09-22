import torch 
import torch.nn as nn 
import torchvision
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator,Generator,initialize_weights


device = torch.device('cuda')
lr = 2e-4
batch_size = 128
image_size = 64
channels = 1
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64


transforms = transforms.Compose([
	transforms.Resize(image_size),
	transforms.ToTensor(),
	transforms.Normalize([0.5 for _ in range(channels)],[0.5 for _ in range(channels)]) ])
dataset = datasets.MNIST(root = 'dataset/',train=True,transform=transforms,download=True)

dataloader = DataLoader(dataset,batch_size = batch_size,shuffle=True)
gen = Generator(z_dim,channels,features_gen).to(device)
disc = Discriminator(channels,features_disc).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr=lr,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=lr,betas=(0.5,0.999))
criterion= nn.BCELoss()

fixed_noise = torch.randn(32,z_dim,1,1).to(device)
writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0

gen.train() 
disc.train()

for epoch in tqdm(range(num_epochs)):
	for batch_idx,(real,_) in enumerate(dataloader):
		real = real.to(device)
		noise = torch.randn((batch_size,z_dim,1,1)).to(device)
		fake = gen(noise)


		#train disc
		disc_real = disc(real).reshape(-1)
		loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
		disc_fake = disc(fake).reshape(-1)
		loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
		loss_disc = (loss_disc_real + loss_disc_fake) /2
		print(loss_disc)
		disc.zero_grad()
		loss_disc.backward(retain_graph=True)
		opt_disc.step()

		#train gen
		output = disc(fake).reshape(-1)
		loss_gen = criterion(output,torch.ones_like(output))
		gen.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		if batch_idx % 100 == 0:
			with torch.no_grad():
				fake = gen(fixed_noise)

				img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
				img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)

				writer_real.add_image('real',img_grid_real,global_step=step)
				writer_fake.add_image('fake',img_grid_fake,global_step=step)

			step +=1