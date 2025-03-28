
import os
import pathlib
import random
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from horama.plots import plot_maco
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torchvision.ops import roi_align
from tqdm import tqdm


def worker_init_fn(worker_id):
    random.seed(worker_id)

"""
common.pyの中身
"""
def standardize(tensor):
    # standardizes the tensor to have 0 mean and unit variance
    tensor = tensor - torch.mean(tensor)
    tensor = tensor / (torch.std(tensor) + 1e-4)
    return tensor


@lru_cache(maxsize=8)
def get_color_correlation_svd_sqrt(device):
    return torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32, device=device
    )

"""
FDSLモデルの場合
"""
# @lru_cache(maxsize=8)
# def get_color_correlation_svd_sqrt(device):
#     return torch.tensor(
#         [[0.0743, -0.0872, 0.0998],
#          [0.0385, 0.1081, 0.0996],
#          [-0.1126, -0.0206, 0.0999]],
#         dtype=torch.float32, device=device
#     )


def recorrelate_colors(image, device):
    # recorrelates the colors of the images
    assert len(image.shape) == 3

    # tensor for color correlation svd square root
    color_correlation_svd_sqrt = get_color_correlation_svd_sqrt(device)

    permuted_image = image.permute(1, 2, 0).contiguous()
    flat_image = permuted_image.view(-1, 3)

    recorrelated_image = torch.matmul(flat_image, color_correlation_svd_sqrt)
    recorrelated_image = recorrelated_image.view(permuted_image.shape).permute(2, 0, 1)

    return recorrelated_image


def optimization_step(objective_function, image, box_size, noise_level,
                      number_of_crops_per_iteration, model_input_size):
    # performs an optimization step on the generated image
    # pylint: disable=C0103
    assert box_size[1] >= box_size[0]
    assert len(image.shape) == 3

    # print(image.shape) #torch.Size([3, 1280, 1280])

    device = image.device
    image.retain_grad()

    # generate random boxes
    x0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    y0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    delta_x = torch.rand((number_of_crops_per_iteration,),
                         device=device) * (box_size[1] - box_size[0]) + box_size[1]
    delta_y = delta_x

    boxes = torch.stack([torch.zeros((number_of_crops_per_iteration,), device=device),
                         x0 - delta_x * 0.5,
                         y0 - delta_y * 0.5,
                         x0 + delta_x * 0.5,
                         y0 + delta_y * 0.5], dim=1) * image.shape[1]

    cropped_and_resized_images = roi_align(image.unsqueeze(
        0), boxes, output_size=(model_input_size, model_input_size)).squeeze(0)
    
    # print(cropped_and_resized_images.shape) #torch.Size([6, 3, 224, 224])

    # add normal and uniform noise for better robustness
    cropped_and_resized_images.add_(torch.randn_like(cropped_and_resized_images) * noise_level)
    cropped_and_resized_images.add_(
        (torch.rand_like(cropped_and_resized_images) - 0.5) * noise_level)

    # compute the score and loss
    score = objective_function(cropped_and_resized_images)
    loss = -score

    # print(image.shape) #torch.Size([3, 1280, 1280])

    return loss, image

"""
maco_fv.pyの中身
"""

MACO_SPECTRUM_URL = ("https://storage.googleapis.com/serrelab/loupe/"
                     "spectrums/imagenet_decorrelated.npy")
MACO_SPECTRUM_FILENAME = 'spectrum_decorrelated.npy'

"""
Datasetに合わせた実装
"""

def init_maco_buffer(image_shape, std_deviation=1.0, dataloader=0):
    """MACO バッファを、ランダムな位相とマグニチュードテンプレートで初期化します。
    dataloader が指定された場合は、データセット中の画像に対して FFT を適用し、
    その平均のマグニチュード（絶対値）を計算します。
    指定がない場合は、事前に用意された（ダウンロードした）マグニチュードテンプレートを使用します。

    Args:
        image_shape (tuple): 画像の形状、例：(channels, height, width)。
        std_deviation (float): ランダム位相生成時の標準偏差。
        dataloader (torch.utils.data.DataLoader, optional): 
            マグニチュードをデータセットから計算する場合のデータローダ。
            指定しない場合はダウンロード済みテンプレートを使用します。
        device (torch.device, optional): dataloader を用いる際の FFT 計算用デバイス。

    Returns:
        tuple: (magnitude, random_phase)
            - magnitude: shape が (channels, height, width//2+1) のテンソル。平均 FFT マグニチュード。
            - random_phase: 同じ shape のランダム位相テンソル。
    """
    # FFT の結果として得られる one-sided spectrum の形状
    spectrum_shape = (image_shape[0], image_shape[1] // 2 + 1)
    
    # ランダム位相バッファの生成
    random_phase = torch.randn(3, *spectrum_shape, dtype=torch.float32) * std_deviation

    if dataloader == 0:
        magnitude_path = pathlib.Path(
            "outputs/magnitude_FDSL_test.pth"
        )
        if magnitude_path.exists():
            print("dataloader 確認！保存済みマグニチュードをロードします🤘")
            magnitude = torch.load(magnitude_path)
        else:
            print("dataloader 指定！データセットから FFT 平均マグニチュード計算開始🤘")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])

            # val_transform = transforms.Compose([transforms.Resize((224,224), interpolation=2),
            #                                     transforms.ToTensor(), normalize])
            # train_transform = transforms.Compose([transforms.RandomCrop((224,224)),
            #     transforms.ToTensor(), normalize])
            test_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            dataset = datasets.ImageFolder("data/FractalDB-1k", transform=test_transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                                        num_workers=8, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
            # 画像のチャネル数は最初のサンプルから取得
            sample_img, _ = dataset[0]
            channels = sample_img.shape[0]

            # ランダム位相の初期化（TF の tf.random.normal と同じ感じ）
            random_phase = torch.randn(channels, *spectrum_shape, dtype=torch.float32, device=device) * std_deviation

            magnitude_sum = 0.0
            count = 0
            for images, _ in tqdm(dataloader):
                images = images.to(device)  # images shape: [N, C, H, W]
                # FFT は最後の2次元に対して実行（rfft2）
                fft_images = torch.fft.rfft2(images, norm="backward")
                magnitudes = torch.abs(fft_images)  # shape: [N, C, H, W_rfft]
                magnitude_sum += magnitudes.sum(dim=0)  # sum over batch, result shape: [C, H, W_rfft]
                count += images.size(0)
            magnitude_mean = magnitude_sum / count  # [C, H, W_rfft]
            
            # TF の場合は、画像サイズが異なるとき resize しているので、同じように補間で調整する
            # F.interpolate は 4D 入力だから unsqueeze(0) して使う
            magnitude_mean = magnitude_mean.unsqueeze(0)  # [1, C, H, W_rfft]
            magnitude_resized = F.interpolate(
                magnitude_mean, size=spectrum_shape, mode="bilinear", align_corners=False
            )
            magnitude = magnitude_resized.squeeze(0)  # [C, spectrum_shape[0], spectrum_shape[1]]
            torch.save(magnitude, magnitude_path)

    else:
        # dataloader が指定されていない場合は、ダウンロード済みのテンプレートを使用
        if not os.path.isfile(MACO_SPECTRUM_FILENAME):
            download_url(MACO_SPECTRUM_URL, root=".", filename=MACO_SPECTRUM_FILENAME)
        magnitude = torch.tensor(np.load(MACO_SPECTRUM_FILENAME), dtype=torch.float32)
        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=spectrum_shape,
            mode='bilinear',
            align_corners=False,
            antialias=True
        )[0]

    return magnitude, random_phase


def maco_preconditioner(magnitude_template, phase, values_range, device):
    # apply the maco preconditioner to generate spatial images from magnitude and phase
    # tfel: check why r exp^(j theta) give slighly diff results
    standardized_phase = standardize(phase)
    complex_spectrum = torch.complex(torch.cos(standardized_phase) * magnitude_template,
                                     torch.sin(standardized_phase) * magnitude_template)

    # transform to spatial domain and standardize
    spatial_image = torch.fft.irfft2(complex_spectrum)
    spatial_image = standardize(spatial_image)

    # recorrelate colors and adjust value range
    color_recorrelated_image = recorrelate_colors(spatial_image, device)
    final_image = torch.sigmoid(
        color_recorrelated_image) * (values_range[1] - values_range[0]) + values_range[0]
    return final_image


def maco_fdsl(objective_function, total_steps=1000, learning_rate=1.0, image_size=1280,
         model_input_size=224, noise=0.05, values_range=(-2.5, 2.5),
         crops_per_iteration=6, box_size=(0.20, 0.25),
         device='cuda'):
    # perform the maco optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    magnitude, phase = init_maco_buffer(
        (image_size, image_size), std_deviation=1.0)
    magnitude = magnitude.to(device)
    phase = phase.to(device)
    phase.requires_grad = True

    optimizer = torch.optim.NAdam([phase], lr=learning_rate)
    transparency_accumulator = torch.zeros(
        (3, image_size, image_size)).to(device)

    for _ in tqdm(range(total_steps)):
        optimizer.zero_grad()

        # preprocess and compute loss
        img = maco_preconditioner(magnitude, phase, values_range, device)
        loss, img = optimization_step(
            objective_function, img, box_size, noise, crops_per_iteration, model_input_size)

        loss.backward()
        # get dL/dx to update transparency mask
        transparency_accumulator += torch.abs(img.grad)
        optimizer.step()

    final_image = maco_preconditioner(magnitude, phase, values_range, device)

    return final_image, transparency_accumulator

"""
ここのnormalize関数がhorama_starter_fdsl.pyのものと違う
"""

# def numpy_normalize(image, mean, std):
#     # image: [H, W, C] を想定
#     mean = np.array(mean).reshape(1, 1, -1)
#     std = np.array(std).reshape(1, 1, -1)
#     return (image - mean) / std


# def plot_maco(image, alpha, percentile_image=1.0, percentile_alpha=80):
#     # visualize image with alpha mask overlay after normalization and clipping
#     image, alpha = check_format(image), check_format(alpha)
#     image = clip_percentile(image, percentile_image)
#     # normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
#     # image = normalize(image)
#     normalized_image = numpy_normalize(image, [0.2, 0.2, 0.2], [0.5, 0.5, 0.5])

#     # mean of alpha across channels, clipping, and normalization
#     alpha = np.mean(alpha, -1, keepdims=True)
#     alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
#     alpha = alpha / alpha.max()

#     # overlay alpha mask on the image
#     plt.imshow(np.concatenate([normalized_image, alpha], -1))
#     plt.axis('off')


if __name__ == "__main__":

    # model = timm.create_model('resnet50.tv_in1k', pretrained=True).cuda().eval()
    
    """
    FDSLモデルを使用
    学習済のモデルの重み：FractalDB-1000_res50.pth
    """
    model = timm.create_model('resnet50', pretrained=False).cuda().eval()
    
    # from FractalDB.resnet import resnet50
    # parser = argparse.ArgumentParser(description='Horama FDSL')
    # parser.add_argument('--numof_classes', type=int, default=1000)
    # args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = resnet50(pretrained=False, num_classes=args.numof_classes).to(device)
    model.load_state_dict(torch.load('weights/FractalDB-1000_resnet50_epoch90.pth'))

    objective = lambda images: torch.mean(model(images)[:, 388])

    # image1, alpha1 = maco_fdsl(objective, values_range=(-2.5, 2.5))

    """
    MACOで構成された画像をモデルに入力して予測
    """
    # percentile_image=1.0
    # image, alpha = check_format(image1), check_format(alpha1)
    # image = clip_percentile(image, percentile_image)
    # image = normalize(image)

    # # print(image.shape)

    # image = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, :, :]
    # image = image.to("cuda")

    # # print(image.shape) #(1, 3, 224, 224) #(1, 3, 1280, 1280)

    # logits = model(image)
    # probabilities = torch.nn.functional.softmax(logits, dim=1)

    # top3_prob, top3_idx = torch.topk(probabilities, k=3, dim=1)

    # print("Top 3 class indices:", top3_idx)
    # print("Top 3 probabilities:", top3_prob)

    # plot_maco(image1, alpha1)
    # plt.title(
    # f'Indices: {top3_idx.squeeze().tolist()}\n'
    # f'Probabilities: {top3_prob.squeeze().tolist()}',
    # fontsize=10  # フォントサイズを小さめに設定
    # )
    # plt.savefig("horama_output.png")  # 画像を保存する
    # plt.close()  # 図を閉じる

    for crop_size in [0.10, 0.25, 0.50, 0.60, 0.80]:
        img, alpha = maco_fdsl(objective, values_range = (-2.5, 2.5), box_size = (crop_size, crop_size))

        # percentile_image=1.0
        # image, alpha = check_format(img), check_format(alpha)
        # image = clip_percentile(image, percentile_image)
        normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])

        # もしimgがTensorならPIL Imageに変換
        if isinstance(img, torch.Tensor):
            print("Tensor to PIL Image")
            PIL_img = transforms.ToPILImage()(img)
        test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        image = test_transform(PIL_img).unsqueeze(0)

        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, :, :]
        image = image.to("cuda")

        logits = model(image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        top3_prob, top3_idx = torch.topk(probabilities, k=3, dim=1)

        plot_maco(img, alpha)
        plt.title(
        f'Crop size: {crop_size*100.0}%\n'
        f'Indices: {top3_idx.squeeze().tolist()}\n'
        f'Probabilities: {top3_prob.squeeze().tolist()}',
        fontsize=10  # フォントサイズを小さめに設定
        )
        plt.savefig(f"outputs/resnet50/FDSL/388_crop_size_{crop_size}.png")  # 画像を保存する
        plt.close()  # 図を閉じる
