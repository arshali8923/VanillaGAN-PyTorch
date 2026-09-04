import torchvision
import os
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms
from PIL import Image

#image load => transform => dataset out of all img

class ImageProcessor:
    def __init__(self , root_dir_path , transformations = None):
        self.root_dir_path = root_dir_path
        self.transformations = transformations

        # list of path for all images
        self.all_img_path =  [os.path.join(root_dir_path , img) for img in os.listdir(root_dir_path)]

    def __len__(self):
        return len(self.all_img_path)

    def __getitem__(self , idx):
        img_path = self.all_img_path[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transformations:
            img = self.transformations(img)

        return img

root_dir_path = r"C:\Users\arsha\Desktop\prime ai&ml\GAN\img_align_celeba\img_align_celeba"

transformations = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))  #[-1,1]
])

dataset = ImageProcessor(root_dir_path , transformations)
print(f"loaded {len(dataset)} images")

dataloader = DataLoader(dataset , batch_size = 128 , shuffle = True)

"""Generator Network"""

import torch.nn as nn
import torch.optim as optim
import numpy as np

class Generator(nn.Module):
    def __init__(self , z_dim = 100 , img_channel = 3):
        super(Generator , self).__init__()

        #fully connected (dense) layer

        self.model = nn.Sequential(
            nn.Linear(z_dim , 256),  #100 => 256
            nn.ReLU(),

            nn.Linear(256 , 512),
            nn.ReLU(),

            nn.Linear(512 , 1024),
            nn.ReLU(),

            nn.Linear(1024 , 64 * 64 * img_channel),
            nn.Tanh()   #[-1,1]  #take coz we match the real img structured to fake images
        )   

    def forward(self , z):
        img = self.model(z)
        img = img.view(img.size(0) , 3 , 64 , 64)   #.size return batch size  , combined o/p convert to image dimension 
        return img

"""DESCRIMINATOR"""

class Descriminator(nn.Module):
    def __init__(self , img_channel = 3):
        super(Descriminator , self ).__init__()

        self.model = nn.Sequential(
            nn.flatten(), #4D => 1D

            nn.Linear(img_channel * 64 * 64 , 1024),
            nn.LeakyReLU(0.2 , inplace = True),

            nn.Linear(1024 , 512),
            nn.LeakyReLU(0.2 , inplace = True),

            nn.Linear(512 , 256),
            nn.LeakyReLU(0.2 , inplace = True),

            nn.Linear(256 , 1),
            nn.Sigmoid()   #probability of being real/fake
        )

        def forward(self , img):
            return self.model(img)


GAN_loss = nn.BCELoss()

generator = Generator()
g_optimizer = optim.Adam(generator.parameters() , lr = 0.0002 , betas = (0.5 , 0.999))

discriminator = Descriminator()
d_optimizer = optim.Adam(discriminator.parameters() , lr = 0.0002 , betas = (0.5 , 0.999))

"""Device"""

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("gpu")

else:
    device = torch.device("cpu")

#shift to the device

generator = generator.to(device)
discriminator = discriminator.to(device)


"""TRAIN LOADER"""
def train(generator , discriminator , dataloader , epochs = 10):

    for epoch in range(epochs):
        for i , imgs in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            #create real img label and fake img label
            real_labels = torch.ones(batch_size , 1).to(device)
            fake_labels = torch.zeros(batch_size , 1).to(device)

            #train the descriminators
            d_optimizer.zero_grad()

            fake_imgs = generator(torch.randomn(batch_size , 100)).to(device)

            real_loss = GAN_loss(discriminator(real_imgs) , real_labels)
            fake_loss = GAN_loss(discriminator(fake_imgs.detach())  , fake_labels)

            d_loss = (real_loss + fake_loss)/2

            d_loss.backward()
            d_optimizer.step()

            #train the generators

            g_optimizer.zero_grad()

            g_loss = GAN_loss(discriminator(fake_imgs) , real_labels)

            g_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                print(f"for epoch : {epoch + 1}/{epochs}... batch {i+1}...G_loss {g_loss}... D_loss {d_loss}")

    #save generated images for each epoch
    save_generated_images(generator , epoch , device)

import matplotlib.pyplot as plt
import torchvision

def save_generated_images(generator , epoch , device , num_imgs = 8):
    z = torch.randn(num_imgs , 100).to(device)
    generated_imgs = generator(z).detach().cpu()

    #imge generate [-1,1] but expect in rgb [0,1] so normalize = true
    grid = torchvision.utils.make_grid(generated_imgs , nrow = 4 , normalize = True)

    plt.imshow(np.transpose(grid , (1 , 2, 0)))
    plt.title(f"epoch {epoch+1}")
    plt.axis("off")
    plt.savefig(f"epoch_{epoch+1}.png")
    plt.show()

train(generator , discriminator , dataloader , epochs = 10)
