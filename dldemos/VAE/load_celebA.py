"""
处理数据;
os用来处理文件路径、列目录文件名;
DataLoader定义怎么按batch取样本;
transforms做图像预处理, 比如裁剪,缩放,转tensor;
Image读取图片文件;
"""

import os
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CelebADataset(Dataset):
    """
    自定义的数据集类, 需要实现__init__, __len__, __getitem__;
    """
    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    # 返回数据集图片数;
    def __len__(self) -> int:
        return len(self.filenames)

    # 给一个索引index, 返回第index张图片处理后的结果;
    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


def get_dataloader(root='/data/workdir/panwei/Data/CIFAR10/img_align_celeba', **kwargs):
    dataset = CelebADataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True)



if __name__ == '__main__':
    """
    把一个批次的图片拼成一张输出;
    """
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    # 取出第一个batch, [16, 3, 64, 64];
    print(img.shape)
    # Concat 4x4 images
    N, C, H, W = img.shape
    assert N == 16
    img = th.permute(img, (1, 0, 2, 3))
    img = th.reshape(img, (C, 4, 4 * H, W))
    img = th.permute(img, (0, 2, 1, 3))
    img = th.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.save('/data/workdir/panwei/Data/CIFAR10/tmp.jpg')







