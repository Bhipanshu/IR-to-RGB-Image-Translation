import argparse
import itertools
from torch.autograd import Variable
from PIL import Image
import torch
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
import torchvision
import os

from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
# parser.add_argument('--dataroot', type=str, default='datasets/wheatData_aligned_croppedResized/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=1, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
# parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)


# Dataset loader
transform_ir = torchvision.transforms.Compose([
    torchvision.transforms.Resize((240,320)),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor()
])

transform_rgb = torchvision.transforms.Compose([
    torchvision.transforms.Resize((240,320)),
    torchvision.transforms.ToTensor()
])

root_dir = './RGB-IRPair'


class IR_Dataset(Dataset):
    def __init__(self, root_dir_ir, transform=None):
        self.root_dir_ir = root_dir_ir
        self.transform = transform
        self.ir_images = []

        for root, dirs, files in os.walk(root_dir_ir):
            for dir in dirs:
                if "IR" in dir:
                    self.ir_images += [os.path.join(root, dir, file) for file in os.listdir(os.path.join(root, dir))]

    def __len__(self):
        return len(self.ir_images)
    
    def __getitem__(self, idx):
        ir_image = Image.open(self.ir_images[idx])
        if self.transform:
            ir_image = self.transform(ir_image)
        return ir_image
    

class RGB_Dataset(Dataset):
    def __init__(self, root_dir_rgb, transform=None):
        self.root_dir_rgb = root_dir_rgb
        self.transform = transform
        self.rgb_images = []

        for root, dirs, files in os.walk(root_dir_rgb):
            for dir in dirs:
                if "RGB" in dir:
                    self.rgb_images += [os.path.join(root, dir, file) for file in os.listdir(os.path.join(root, dir))]

    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_images[idx])
        if self.transform:
            rgb_image = self.transform(rgb_image)
        return rgb_image
    
    
dataset_ir = IR_Dataset(root_dir,transform=transform_ir)
print(len(dataset_ir))
dataset_rgb = RGB_Dataset(root_dir,transform=transform_rgb)
print(len(dataset_rgb))

train_size = int(0.8 * len(dataset_ir))
test_size = len(dataset_ir) - train_size

train_dataset_ir = torch.utils.data.Subset(dataset_ir, list(range(train_size)))
test_dataset_ir = torch.utils.data.Subset(dataset_ir, list(range(train_size, len(dataset_ir))))

train_dataset_rgb = torch.utils.data.Subset(dataset_rgb, list(range(train_size)))
test_dataset_rgb = torch.utils.data.Subset(dataset_rgb, list(range(train_size, len(dataset_rgb))))


class CombinedDataset(Dataset):
    def __init__(self,dataset_ir,dataset_rgb):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb

    def __len__(self):
        return max(len(self.dataset_ir), len(self.dataset_rgb))

    def __getitem__(self, index):
        item1 = self.dataset_ir[index % len(self.dataset_ir)]
        item2 = self.dataset_rgb[index % len(self.dataset_rgb)]
        return (item1, item2)
    
combined_dataset = CombinedDataset(train_dataset_ir,train_dataset_rgb)
test_dataset = CombinedDataset(test_dataset_ir,test_dataset_rgb)


train_loader = DataLoader(combined_dataset, batch_size=opt.batchSize, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


netG_A2B = Generator(opt.input_nc, opt.output_nc)

netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


state_dict = torch.load('best_netG_A2B.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] 
    new_state_dict[name] = v
netG_A2B.load_state_dict(new_state_dict)


state_dict = torch.load('best_netG_B2A.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] 
    new_state_dict[name] = v
netG_B2A.load_state_dict(new_state_dict)


state_dict = torch.load('best_netD_A.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' prefix
    new_state_dict[name] = v
netD_A.load_state_dict(new_state_dict)


state_dict = torch.load('best_netD_B.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] 
    new_state_dict[name] = v
netD_B.load_state_dict(new_state_dict)


netG_A2B = torch.nn.DataParallel(netG_A2B,device_ids=[0,1])
netG_B2A = torch.nn.DataParallel(netG_B2A,device_ids=[0,1])
netD_A = torch.nn.DataParallel(netD_A,device_ids=[0,1])
netD_B = torch.nn.DataParallel(netD_B,device_ids=[0,1])
netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)
# # Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor

input_A = Tensor(opt.batchSize, opt.input_nc, 240,320)
input_B = Tensor(opt.batchSize, opt.output_nc, 240,320)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


# Loss plot
logger = Logger(opt.n_epochs, len(train_loader))

best_loss = 1000000
###################################
from tqdm import tqdm
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    # with tqdm(range(len(train_loader))) as t:
    for i, (data1,data2) in enumerate(train_loader):
        
        img1_ir = data1.to(device)
        img1_rgb = data2.to(device)
            # print(img1_ir.shape)

        # Set model input

        if img1_ir.shape[0] != opt.batchSize:
            continue
        real_A = Variable(input_A.copy_(img1_ir))
        real_B = Variable(input_B.copy_(img1_rgb))



        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()


        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        print(f'Epoch: {epoch}, Batch: {i}, Loss_G: {loss_G}, Loss_D: {loss_D_A + loss_D_B}',end='\r')
    print(f'Epoch: {epoch}, Loss_G: {loss_G}, Loss_D: {loss_D_A + loss_D_B}')


        # Save images
    torchvision.utils.save_image(real_A, f'output/real_A_continue/real_A_{epoch}_{i}.png')
    torchvision.utils.save_image(real_B, f'output/real_B_continue/real_B_{epoch}_{i}.png')
    torchvision.utils.save_image(fake_A, f'output/fake_A_continue/fake_A_{epoch}_{i}.png')
    torchvision.utils.save_image(fake_B, f'output/fake_B_continue/fake_B_{epoch}_{i}.png')
        

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    if loss_G < best_loss:
        best_loss = loss_G
        torch.save(netG_A2B.state_dict(), 'output/best_netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/best_netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/best_netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/best_netD_B.pth')
