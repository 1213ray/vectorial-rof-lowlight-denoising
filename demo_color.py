"""
實驗 2、3：彩色影像的純量 TV 與向量 TV（VTV）比較。

模擬低光雜訊：在乾淨影像加上高斯雜訊，並把紅色通道的雜訊放大（--red-bias），
製造低光拍攝常見的色度雜訊；接著分別以純量 TV 與 VTV 去噪，比較 PSNR / SSIM 與局部細節。
有 CuPy 與 NVIDIA GPU 時自動用 GPU 計算，否則用 NumPy（較慢）。

用法：
    python demo_color.py --image images/flags.jpg      --out results/flags
    python demo_color.py --image images/open_sign.jpg  --out results/open_sign
    python demo_color.py --image images/flags.jpg --cpu --iters 300
"""
import argparse
import os
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from rof import cp, rof_denoise


def run(img_noisy, weight, iters, mode, use_gpu):
    """執行去噪並計時；use_gpu 時在 GPU 上算完再傳回 CPU。"""
    start = time.time()
    if use_gpu:
        u = rof_denoise(cp.asarray(img_noisy), weight=weight, n_iter=iters, mode=mode)
        cp.cuda.Stream.null.synchronize()
        u = cp.asnumpy(u)
    else:
        u = rof_denoise(img_noisy, weight=weight, n_iter=iters, mode=mode)
    print(f'  {mode:9s} TV（weight={weight}, iters={iters}）耗時 {time.time() - start:.2f} 秒')
    return np.clip(u, 0, 1)


def main():
    parser = argparse.ArgumentParser(description='純量 TV vs 向量 TV')
    parser.add_argument('--image', default='images/open_sign.jpg', help='輸入影像')
    parser.add_argument('--out', default='results/open_sign', help='輸出資料夾')
    parser.add_argument('--sigma', type=float, default=0.125, help='基礎雜訊標準差')
    parser.add_argument('--red-bias', type=float, default=3, help='紅色通道雜訊放大倍數（模擬色度雜訊）')
    parser.add_argument('--weight-scalar', type=float, default=0.15, help='純量 TV 權重')
    parser.add_argument('--weight-vector', type=float, default=0.22, help='向量 TV 權重')
    parser.add_argument('--iters', type=int, default=1000, help='迭代次數')
    parser.add_argument('--crop', type=float, nargs=4, default=[0.4, 0.6, 0.4, 0.6],
                        metavar=('Y0', 'Y1', 'X0', 'X1'), help='局部放大區域（比例 0–1）')
    parser.add_argument('--cpu', action='store_true', help='強制使用 CPU')
    parser.add_argument('--save-full', action='store_true', help='另存全尺寸的雜訊影像與兩種結果')
    args = parser.parse_args()

    use_gpu = cp is not None and not args.cpu
    print(f'使用 {"GPU (CuPy)" if use_gpu else "CPU (NumPy)"}')

    img_clean = img_as_float(imread(args.image)).astype(np.float32)
    if img_clean.ndim == 3 and img_clean.shape[2] == 4:      # 去掉 alpha 通道
        img_clean = img_clean[:, :, :3]
    h, w, _ = img_clean.shape
    print(f'影像：{args.image}（{w}×{h}）')

    # 模擬低光雜訊：紅色通道雜訊較強
    np.random.seed(42)
    noise = np.random.randn(*img_clean.shape).astype(np.float32) * args.sigma
    noise[:, :, 0] *= args.red_bias
    img_noisy = np.clip(img_clean + noise, 0, 1)

    img_scalar = run(img_noisy, args.weight_scalar, args.iters, 'scalar', use_gpu)
    img_vector = run(img_noisy, args.weight_vector, args.iters, 'vectorial', use_gpu)

    def metrics(img):
        return (psnr(img_clean, img, data_range=1),
                ssim(img_clean, img, data_range=1, channel_axis=2))

    scores = {name: metrics(img) for name, img in
              [('Noisy', img_noisy), ('Scalar TV', img_scalar), ('Vectorial TV', img_vector)]}
    for name, (p, s) in scores.items():
        print(f'{name:13s} PSNR {p:.2f} dB, SSIM {s:.3f}')

    os.makedirs(args.out, exist_ok=True)
    y0, y1 = int(h * args.crop[0]), int(h * args.crop[1])
    x0, x1 = int(w * args.crop[2]), int(w * args.crop[3])
    crops = {name: img[y0:y1, x0:x1] for name, img in
             [('noisy', img_noisy), ('scalar', img_scalar), ('vectorial', img_vector)]}
    for name, crop in crops.items():
        imsave(f'{args.out}/zoom_{name}.png', img_as_ubyte(crop))
    if args.save_full:
        imsave(f'{args.out}/full_noisy.png', img_as_ubyte(img_noisy))
        imsave(f'{args.out}/full_scalar.png', img_as_ubyte(img_scalar))
        imsave(f'{args.out}/full_vectorial.png', img_as_ubyte(img_vector))

    # 對比圖：上排全圖（紅框為放大區域），下排局部放大
    panel_w = 5
    fig, ax = plt.subplots(2, 3, figsize=(3 * panel_w, 2 * panel_w * h / w + 1.4))
    full = [img_noisy, img_scalar, img_vector]
    for a, img, (name, (p, s)) in zip(ax[0], full, scores.items()):
        a.imshow(img)
        a.set_title(f'{name}\nPSNR {p:.2f} dB, SSIM {s:.3f}')
        a.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      linewidth=2, edgecolor='#e34948', facecolor='none'))
    for a, (name, crop) in zip(ax[1], crops.items()):
        a.imshow(crop, interpolation='nearest')
        a.set_title(f'Zoom: {name}')
    for a in ax.flat:
        a.axis('off')
    fig.tight_layout()
    path = f'{args.out}/comparison.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'已儲存：{path}')


if __name__ == '__main__':
    main()
