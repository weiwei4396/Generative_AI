
from time import time
import os
import torch as th
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from model_vae import VAE
from load_celebA import get_dataloader

lr = 0.005
kl_weight = 0.00025
n_epochs = 10

def loss_fn(y, y_hat, mean, logvar):
    # B*C*H*W 所有的元素都MSE;
    recons_loss = F.mse_loss(y_hat, y)
    # [B, latent_dim] -> [B];
    # -0.5 * th.sum 是每个样品的KL; th.mean是对batch所有样品求平均;
    kl_loss = th.mean(-0.5 * th.sum(1 + logvar - mean**2 - th.exp(logvar), 1), 0)
    # 加权版的loss;
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model):
    optimizer = th.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)
    begin_time = time()

    for i in range(n_epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {loss_sum} {minute}:{second}')        
        th.save(model.state_dict(), '/data/workdir/panwei/Data/CIFAR10/base_VAE/model.pth')


# 从dataloader里取一张真实图片, 送进VAE重建, 再把重建图和原图拼在一起保存;
def reconstruct(device, dataloader, model):
    model.eval()
    # batch.shape -> [16, 3, 64, 64]
    batch = next(iter(dataloader))
    # x -> [1, 3, 64, 64]
    x = batch[0:1, ...].to(device)
    # output是decoder的输出, output -> [1, 3, 64, 64]
    output = model(x)[0]
    # output -> [3, 64, 64]
    output = output[0].detach().cpu()
    input = batch[0].detach().cpu()
    # combined -> [3, 128, 64]
    combined = th.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('/data/workdir/panwei/Data/CIFAR10/base_VAE/tmp_reconstruct.jpg')


# 让VAE随机生成一张新图, 然后保存成图片文件;
def generate(device, model):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('/data/workdir/panwei/Data/CIFAR10/base_VAE/tmp_new.jpg')



def main():
    device = 'cuda:0'
    dataloader = get_dataloader()
    model = VAE().to(device)

    ckpt_path = '/data/workdir/panwei/Data/CIFAR10/base_VAE/model.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(th.load(ckpt_path, map_location='cuda:0'))
        print('Checkpoint loaded.')
    else:
        print('Checkpoint not found, train from scratch.')

    train(device, dataloader, model)
    reconstruct(device, dataloader, model)
    generate(device, model)



if __name__ == '__main__':
    main()







