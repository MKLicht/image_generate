import os
import numpy as np
import torch.autograd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

from load_data import CatData

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 128
num_epoch = 200
z_dimension = 100

# prepare dataset
cat_data = CatData(
    './cat_dataset',
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset=cat_data, batch_size=batch_size, shuffle=True
)

######### Discriminator ########
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256, 0.8),
        )
        ds_size = 64 // 2 ** 5
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.adv_layer(x)
        return x

######### Generator ########
class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 256, 16, 16)
        x =self.conv(x)
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

D = discriminator()
G = generator(100, 256*16*16)
D = D.to(device)
G = G.to(device)
D.apply(weights_init_normal)
G.apply(weights_init_normal)

adv_loss = nn.BCELoss()  
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

if not os.path.exists('./cat_result'):
    os.mkdir('./cat_result')

def to_img(x): 
    out = x.view(-1, 3, 64, 64) 
    return out

for epoch in range(num_epoch): 
    for i, img in enumerate(dataloader):
        num_img = img.size(0)  
        real_label = torch.ones(num_img, 1).to(device)
        fake_label = torch.zeros(num_img, 1).to(device) 
        real_img = img.to(device) 
        real_out = D(real_img)     
    
        g_optimizer.zero_grad()
        # generate sample noises
        z = torch.randn(num_img, z_dimension).to(device)  
        fake_img = G(z)  
        fake_out = D(fake_img) 
        g_loss = adv_loss(fake_out, real_label)
        g_loss.backward()  
        g_optimizer.step()
              
        # calculate generator loss
        d_loss_real = adv_loss(real_out, real_label)
        d_loss_fake = adv_loss(D(fake_img.detach()), fake_label) 
        d_loss = d_loss_real + d_loss_fake  
        d_optimizer.zero_grad()  
        d_loss.backward()  
        d_optimizer.step() 

        if (i + 1) % 10 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
                torch.mean(real_out).item(), torch.mean(fake_out).item() 
            ))

        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './cat_result/real_images.png')

        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './cat_result/fake_images-{}.png'.format(epoch + 1))

# save models
if not os.path.exists('./models'):
    os.mkdir('./models')
torch.save(G.state_dict(), './models/generator_DCGAN.pth')
torch.save(D.state_dict(), './models/discriminator_DCGAN.pth')