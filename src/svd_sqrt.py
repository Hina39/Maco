import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


def recalc_color_correlation_svd_sqrt(dataset, device, num_batches=125000):
    """対象データセットからランダムに画像をサンプルして、
    色チャネル（RGB）の共分散行列を計算し、固有値分解で
    各固有値の平方根と固有ベクトルから変換行列を求めるよ。
    
    戻り値:
      transformation_matrix: [3, 3] の行列
    """
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    total_pixels = 0
    sum_pixels = torch.zeros(3, device=device)       # 各チャネルの総和
    sum_outer = torch.zeros(3, 3, device=device)         # 各ピクセルの外積の総和

    print(len(dataloader))
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(dataloader)):
            if i >= num_batches:
                break
            # images shape: [B, 3, H, W]
            img = img.to(device)
            B, C, H, W = img.shape
            # 合計ピクセル数
            num = B * H * W
            total_pixels += num

            # reshapeして [3, N] に　（N = B*H*W）
            pixels = img.view(C, -1)
            
            # バッチごとにRGBの総和と外積の総和を更新
            sum_pixels += pixels.sum(dim=1)
            sum_outer += pixels @ pixels.t()
            # 安全のため、使い終わったバッチは削除してキャッシュをクリア
            del img, pixels
            torch.cuda.empty_cache()

    # 全体の平均
    mean = sum_pixels / total_pixels
    # 共分散行列は E[xxᵀ] - μμᵀ
    cov = (sum_outer / total_pixels) - (mean.unsqueeze(1) @ mean.unsqueeze(0))

    # 共分散行列の固有値分解（対称行列なので）
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    transformation_matrix = eigenvectors @ torch.diag(sqrt_eigenvalues)
    return transformation_matrix

if __name__ == '__main__':
    # このスクリプトを直接実行したときのサンプルコード
    import pathlib
    from timm.data.transforms import MaybeToTensor
    from torchvision import datasets
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
    )
    transform = Compose(
        [
            Resize(
                248,
                interpolation=InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            CenterCrop((224, 224)),
            MaybeToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(
        root=pathlib.Path("data/FractalDB-1k"), transform=transform
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_matrix = recalc_color_correlation_svd_sqrt(dataset, device)
    print(transform_matrix)

