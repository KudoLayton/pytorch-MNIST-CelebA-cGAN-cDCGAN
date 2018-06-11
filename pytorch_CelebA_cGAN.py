import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(2, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 12288)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(12288, 16384)
        self.fc1_2 = nn.Linear(2, 16384)
        self.fc2 = nn.Linear(32768, 2048)
        self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# label preprocess
img_size = 64
onehot = torch.zeros(2, 2)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2)
fill = torch.zeros([2, 2, img_size, img_size])
for i in range(2):
    fill[i, i, :, :] = 1

with open('data/resized_celebA/gender_label.pkl', 'rb') as fp:
    y_gender_ = pickle.load(fp)

y_gender_ = torch.LongTensor(y_gender_).squeeze()

# fixed noise & label
temp_z0_ = torch.randn(4, 100)
temp_z0_ = torch.cat([temp_z0_, temp_z0_], 0)
temp_z1_ = torch.randn(4, 100)
temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

fixed_z_ = torch.cat([temp_z0_, temp_z1_], 0)
fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()

fixed_z_ = fixed_z_.view(-1, 100)
fixed_y_label_ = onehot[fixed_y_]
fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda(), volatile=True), Variable(fixed_y_label_.cuda(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    G.eval()
    #fixed_z = fixed_z_.reshape((reduce(lambda x, y: x * y, list(fixed_z_.size())) / 100, 100))
    #fixed_y_label_ = fixed_y_label_.reshape((reduce(lambda x, y: x * y, list(fixed_y_label_.size())) / 2, 2))
    #y_size = ((reduce(lambda x, y: x * y, list(fixed_y_label_.size())) / 2, 2))
    #new_fixed_y_label_ = fixed_y_label_.reshape(y_size)
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().reshape((3, 64, 64)).transpose(1, 2, 0)+1)/2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_noise_morp(show=False, save=False, path='result.png'):
    source_z_ = torch.randn(10, 100)
    z_ = torch.zeros(100, 100)
    for i in range(5):
        for j in range(10):
            z_[i*20 + j] = (source_z_[i*2+1] - source_z_[i*2]) / 9 * (j+1) + source_z_[i*2]

    for i in range(5):
        z_[i*20+10:i*20+20] = z_[i*20:i*20+10]

    y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)], 0).type(torch.LongTensor).squeeze()
    y_ = torch.cat([y_, y_, y_, y_, y_], 0)
    y_label_ = onehot[y_]
    z_ = z_.view(-1, 100)
    y_label_ = y_label_.view(-1, 2)

    z_, y_label_ = Variable(z_.cuda(), volatile=True), Variable(y_label_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_, y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(img_size, img_size))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10 * 10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
data_dir = 'data/resized_celebA'          # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
dset.imgs.sort()
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)

# network
G = generator()
D = discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
root = 'CelebA_cGAN_results/'
model = 'CelebA_cGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    num_iter = 0
    for x_, y_ in train_loader:
        # train discriminator D
        D.zero_grad()
        
        if isCrop:
            x_ = x_[:, :, 22:86, 22:86]

        mini_batch = x_.size()[0]
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        y_label_ = torch.zeros(mini_batch, 2)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)
        x_ = x_.view(-1, 3 * 64 * 64)
        x_, y_label_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_label_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda()) 
        D_result = D(x_, y_label_).squeeze()

        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100))
        y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor)
        y_label_ = torch.zeros(mini_batch, 2)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

        z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_label_).squeeze()

        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()
        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100))
        y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor)
        y_label_ = torch.zeros(mini_batch, 2)
        y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

        z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

        G_result = G(z_, y_label_)
        D_result = D(G_result, y_label_).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        num_iter += 1

        if (num_iter % 100) == 0:
            print('%d - %d complete!' % ((epoch+1), num_iter))

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

show_noise_morp(save=True, path=root + model + 'warp.png')
