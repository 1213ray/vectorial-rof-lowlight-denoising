"""
實驗 1：灰階影像的標準 ROF 去噪。

在 cameraman 測試影像（scikit-image 內建）加上高斯雜訊，以 ROF 去噪，
計算 PSNR / SSIM，輸出到 results/gray/。

用法：
    python demo_gray.py
    python demo_gray.py --sigma 0.2 --weight 0.15
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, img_as_ubyte
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from rof import rof_denoise


def main():
    parser = argparse.ArgumentParser(description='灰階 ROF 去噪')
    parser.add_argument('--sigma', type=float, default=0.125, help='高斯雜訊標準差（影像值域 0–1）')
    parser.add_argument('--weight', type=float, default=0.1, help='TV 權重 λ')
    parser.add_argument('--iters', type=int, default=200, help='迭代次數')
    parser.add_argument('--out', default='results/gray', help='輸出資料夾')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    img_clean = img_as_float(data.camera())
    np.random.seed(42)   # 與報告相同的雜訊
    img_noisy = np.clip(img_clean + np.random.randn(*img_clean.shape) * args.sigma, 0, 1)

    print(f'ROF 去噪中（sigma={args.sigma}, weight={args.weight}, iters={args.iters}）...')
    img_rof = rof_denoise(img_noisy, weight=args.weight, n_iter=args.iters)
    img_rof = np.clip(img_rof, 0, 1)

    p_noisy, s_noisy = psnr(img_clean, img_noisy, data_range=1), ssim(img_clean, img_noisy, data_range=1)
    p_rof, s_rof = psnr(img_clean, img_rof, data_range=1), ssim(img_clean, img_rof, data_range=1)
    print(f'雜訊影像：PSNR {p_noisy:.2f} dB, SSIM {s_noisy:.3f}')
    print(f'ROF 結果：PSNR {p_rof:.2f} dB, SSIM {s_rof:.3f}')

    suffix = f'sigma{args.sigma}_weight{args.weight}'
    imsave(f'{args.out}/noisy_{suffix}.png', img_as_ubyte(img_noisy))
    imsave(f'{args.out}/rof_{suffix}.png', img_as_ubyte(img_rof))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6))
    panels = [(img_clean, 'Ground truth'),
              (img_noisy, f'Noisy (σ = {args.sigma})\nPSNR {p_noisy:.2f} dB, SSIM {s_noisy:.3f}'),
              (img_rof, f'ROF (λ = {args.weight})\nPSNR {p_rof:.2f} dB, SSIM {s_rof:.3f}')]
    for a, (img, title) in zip(ax, panels):
        a.imshow(img, cmap='gray', vmin=0, vmax=1)
        a.set_title(title)
        a.axis('off')
    fig.tight_layout()
    path = f'{args.out}/comparison_{suffix}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'已儲存：{path}')


if __name__ == '__main__':
    main()
