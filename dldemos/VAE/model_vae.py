
import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    VAE for 64x64 face generation.
    The hidden dimensions can be tuned.
    """
    # 每层的通道数从16递增到256, 隐变量z的维度是128;
    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()

        # encoder
        # 起始通道数3, 图片shape->[64,64];
        prev_channels = 3
        modules = []
        img_length = 64

        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, cur_channels, kernel_size=3, stride=2, padding=1), 
                    nn.BatchNorm2d(cur_channels), 
                    nn.ReLU()
                ) 
            )        
            prev_channels = cur_channels               
            img_length //= 2               

        # 经过5个卷积层
        # 通道数: 3 -> 16 -> 32 -> 64 -> 128 -> 256
        # 空间尺寸: 64 -> 32 -> 16 -> 8 -> 4 -> 2
        self.encoder = nn.Sequential(*modules)
        # 最后的卷积层输出维度: [B, 256 * 2 * 2] -> [B, 1024]
        # 每个维度一个均值, 一个方差
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.latent_dim = latent_dim

        # decoder
        modules = []
        # [B, latent_dim] -> [B, C * H * W]
        self.decoder_projection = nn.Linear(latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)

        # [4, 3, 2, 1] → [256, 128, 64, 32], 倒着采样做逆卷积;
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i], hiddens[i-1], kernel_size=3, 
                                       stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hiddens[i-1]), 
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0], hiddens[0], kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hiddens[0]), 
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        x = self.decoder_projection(z)
        # [B, C * H * W] → [B, C, H, W]
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded, mean, logvar

    # 直接从潜空间生成一个新z, 解码成图片;
    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded


