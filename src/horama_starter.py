import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from horama import fourier, maco, plot_maco
from horama.plots import check_format, clip_percentile, normalize

mapping_to_timm_model = {
    "resnet50": "timm/resnet50.tv_in1k",  # https://huggingface.co/timm/resnet50.tv_in1k
    "resnet50v2": "timm/resnetv2_50.a1h_in1k",  # https://huggingface.co/timm/resnetv2_50.a1h_in1k
    "mobilenet": "timm/mobilenetv3_large_100.ra_in1k",  # https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k
    "inception": "inception_next_small.sail_in1k",  # https://huggingface.co/timm/inception_next_small.sail_in1k
    "vit": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",  # https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
}

def set_size(w,h):
    plt.rcParams["figure.figsize"] = [w,h]

"""
通常のモデルの場合
"""

model = timm.create_model('resnet50.tv_in1k', pretrained=True).cuda().eval()

"""
ImageNet Linf-norm (ResNet50)
ε = 8 / 255
"""
# model = timm.create_model('resnet50', pretrained=False).cuda().eval()
# # Unexpected key(s) in state_dict: "model", "optimizer", "schedule", "epoch". 

# state_dict = torch.load('weights/imagenet_linf_8.pt')["model"]

# new_state_dict = {}
# for k, v in state_dict.items():
#     # Handle module.attacker.model. prefix
#     if k.startswith('module.attacker.model.'):
#         new_key = k[22:]  # Remove 'module.attacker.model.'
#         new_state_dict[new_key] = v

# model.load_state_dict(new_state_dict)

"""
FDSLモデルを使用
学習済のモデルの重み：FractalDB-1000_res50.pth
"""
# model = timm.create_model('resnet50', pretrained=False).cuda().eval()
# model.load_state_dict(torch.load('weights/FractalDB-1000_resnet50_epoch90.pth'))

"""
FractalDBで事前学習済みのモデルにImagenetでFinetune
重み：pt_FractalDB-1000_ft_ImageNet_resnet50_epoch5.pth
"""
# model = timm.create_model('resnet50', pretrained=False).cuda().eval()
# model.load_state_dict(torch.load('weights/pt_FractalDB-1000_ft_ImageNet_resnet50_epoch90.pth'))

"""
Visual atom
"""
# model = timm.create_model('deit_base_patch16_224', num_classes=21000, pretrained=False).cuda().eval()
# with torch.serialization.safe_globals([argparse.Namespace]):
#     state_dict = torch.load('weights/vit_base_with_visualatom_21k.pth.tar', weights_only=True)["state_dict"]

# model.load_state_dict(state_dict)
# model.load_state_dict(torch.load('weights/vit_base_with_visualatom_21k.pth.tar'))

"""
しのださんのモデルsegrcdb
重み：
"""
# # timm の Swin‑Transformer Base を初期化（事前学習は False にする）
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=False).cuda().eval()
# # 重みを読み込む
# checkpoint = torch.load('weights/upernet_swinb_with_segrcdb_1M.pth')
# state_dict = checkpoint['state_dict']  # チェックポイントの形式に合わせる
# model.load_state_dict(state_dict)

objective = lambda images: torch.mean(model(images)[:, 388])

# image1, alpha1 = maco(objective, values_range=(-2.5, 2.5))


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

# top5_prob, top5_idx = torch.topk(probabilities, k=5, dim=1)

# print("Top 5 class indices:", top5_idx)
# print("Top 5 probabilities:", top5_prob)

# plot_maco(image1, alpha1)
# plt.savefig("horama_output.png")  # 画像を保存する
# plt.close()  # 図を閉じる

for crop_size in [0.10, 0.25, 0.50, 0.60, 0.80]:
    img, alpha = maco(objective, values_range = (-2.5, 2.5), box_size = (crop_size, crop_size))

    percentile_image=1.0
    image, alpha = check_format(img), check_format(alpha)
    image = clip_percentile(image, percentile_image)
    image = normalize(image)

    image = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, :, :]
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
    plt.savefig(f"outputs/resnet50/388_crop_size_{crop_size}.png")  # 画像を保存する
    plt.close()  # 図を閉じる