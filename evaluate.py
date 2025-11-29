# 作者：yjh
# 时间：2025/11/3 12:34
import os
import cv2
import lpips
import torch
import argparse
import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
from skimage import color, metrics
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from torchvision.models import Inception_V3_Weights


# 1. 计算SAM
def calculate_sam(image1, image2):
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    eps = 1e-8
    dot = np.sum(img1 * img2, axis=2)
    norm1 = np.linalg.norm(img1, axis=2)
    norm2 = np.linalg.norm(img2, axis=2)
    cos = dot / (norm1 * norm2 + eps)
    cos = np.clip(cos, -1, 1)
    sam = np.mean(np.arccos(cos))
    return np.degrees(sam)


# 2. 计算ERGAS
def calculate_ergas(image1, image2, ratio=1):
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    mean_ref = np.mean(img2, axis=(0, 1))
    rmse = np.sqrt(np.mean((img1 - img2) ** 2, axis=(0, 1)))
    ergas = 100 / ratio * np.sqrt(np.mean((rmse / (mean_ref + 1e-8)) ** 2))
    return ergas


# 3. 计算UIQI
def calculate_uiqi(image1, image2):
    # 此处使用skimage的SSIM作为UIQI的代理，UIQI公式不常见。
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # 确保 win_size 不超过图片大小
    h, w, c = img1.shape
    win_size = 7
    if min(h, w) < win_size:
        win_size = min(h, w) // 2 * 2 + 1  # 取不大于最小边的奇数
    return ssim(img1, img2, channel_axis=2, win_size=win_size)


# 4. 计算QNR
def calculate_qnr(image1, image2):
    sam_val = calculate_sam(image1, image2)
    ergas_val = calculate_ergas(image1, image2)
    return np.exp(-sam_val / 10) * np.exp(-ergas_val / 100)


# 5. BRISQUE指标
def calculate_brisque(image):
    # 1. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. 计算局部均值和方差
    mu = gaussian_filter(gray, sigma=1)
    mu_sq = mu ** 2
    sigma = np.sqrt(np.abs(gaussian_filter(gray ** 2, sigma=1) - mu_sq))

    # 3. 计算 MSCN
    eps = 1e-8
    mscn = (gray - mu) / (sigma + eps)

    # 4. 统计 MSCN 的特征
    alpha = np.mean(np.abs(mscn))
    beta = np.std(mscn)
    # 可以加入邻域对的统计，但这里简化处理
    score = alpha + beta  # 越小越好

    return score


# 6. NIQE指标
def calculate_niqe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-8)

    # 计算局部均值与方差（类似MSCN）
    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray ** 2, (7, 7), 7 / 6) - mu ** 2))
    mscn = (gray - mu) / (sigma + 1)

    # 统计特征
    shape, loc, scale = stats.gennorm.fit(mscn.flatten())
    niqe_score = np.abs(shape) + np.abs(scale)
    return niqe_score  # 越小越自然


# 7. 计算Histogram偏差
def calculate_histogram(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # 归一化，使总和为1
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    diff = np.sum(np.abs(hist1 - hist2))  # L1距离
    score = 1 - diff / 2
    score = np.clip(score, 0, 1)  # 防止负值
    return score


# 8. Spectral Curve Deviations (基于光谱曲线的偏差)
def calculate_spectral_curve_deviation(image1, image2):
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    eps = 1e-8
    # 计算每个通道的平均反射率
    mean_spec1 = np.mean(img1, axis=(0, 1))
    mean_spec2 = np.mean(img2, axis=(0, 1))
    deviation = np.sqrt(np.mean(((mean_spec1 - mean_spec2) / (mean_spec1 + eps)) ** 2))
    return 1 - deviation  # 越接近1表示光谱一致性越好


# 9. 计算PSNR
def calculate_psnr(image1, image2):
    mse = mean_squared_error(image1.flatten(), image2.flatten())
    return 20 * np.log10(255.0 / np.sqrt(mse))


# 10. 计算SSIM
def calculate_ssim(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # 确保 win_size 不超过图片大小
    h, w, c = img1.shape
    win_size = 7
    if min(h, w) < win_size:
        win_size = min(h, w) // 2 * 2 + 1  # 取不大于最小边的奇数
    return metrics.structural_similarity(image1, image2, channel_axis=2, win_size=win_size)


# 11. 计算CIEDE2000
def calculate_ciede2000(img1, img2):
    """CIEDE2000 色差，取值越小表示越接近"""
    lab1 = color.rgb2lab(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    lab2 = color.rgb2lab(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    return np.mean(delta_e)

# 12. 计算LPIPS
def calculate_lpips(img1, img2, lpips_model=None):
    """LPIPS 感知距离，越小越好"""
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex')
    img1_t = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    img2_t = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    with torch.no_grad():
        d = lpips_model(img1_t, img2_t)
    return d.item()


# 13. 计算FID
def calculate_fid(imgs1, imgs2):
    """计算FID (Fréchet Inception Distance)，越小越好"""
    from torchvision.models import inception_v3
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # 只保留特征部分
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def get_features(imgs):
        feats = []
        with torch.no_grad():
            for img in imgs:
                img_t = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                feat = model(img_t).cpu().numpy().reshape(-1)
                feats.append(feat)
        return np.array(feats)

    act1 = get_features(imgs1)
    act2 = get_features(imgs2)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# 图片加载和对齐（考虑命名不同但目录一致）
def load_images_from_folder(folder, file_names):
    images = []
    for filename in file_names:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def resize_to_match(img1, img2):
    h, w = img2.shape[:2]
    return cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)



def main(train_folder, target_folder):
    train_images = sorted(os.listdir(train_folder))
    target_images = sorted(os.listdir(target_folder))

    sam, ergas, uiqi, qnr, brisque, niqe, hist, spectral, psnr, ssim, ciede, lpips_list = [], [], [], [], [], [], [], [], [], [], [], []

    lpips_model = lpips.LPIPS(net='alex')

    imgs1, imgs2 = [], []

    for train_img, target_img in zip(train_images, target_images):
        train_image = cv2.imread(os.path.join(train_folder, train_img))
        target_image = cv2.imread(os.path.join(target_folder, target_img))

        # ✅ 确保尺寸一致
        if train_image.shape != target_image.shape:
            train_image = resize_to_match(train_image, target_image)

        sam.append(calculate_sam(train_image, target_image))
        ergas.append(calculate_ergas(train_image, target_image))
        uiqi.append(calculate_uiqi(train_image, target_image))
        qnr.append(calculate_qnr(train_image, target_image))
        brisque.append(calculate_brisque(train_image))
        niqe.append(calculate_niqe(train_image))
        hist.append(calculate_histogram(train_image, target_image))
        spectral.append(calculate_spectral_curve_deviation(train_image, target_image))
        psnr.append(calculate_psnr(train_image, target_image))
        ssim.append(calculate_ssim(train_image, target_image))
        ciede.append(calculate_ciede2000(train_image, target_image))
        lpips_list.append(calculate_lpips(train_image, target_image, lpips_model))

        # FID 收集图像
        imgs1.append(train_image)
        imgs2.append(target_image)

    print(f"SAM: {np.mean(sam):.3f}")
    print(f"ERGAS: {np.mean(ergas):.3f}")
    print(f"UIQI: {np.mean(uiqi):.3f}")
    print(f"QNR: {np.mean(qnr):.3f}")
    print(f"BRISQUE: {np.mean(brisque):.3f}")
    print(f"NIQE: {np.mean(niqe):.3f}")
    print(f"Histogram: {np.mean(hist):.3f}")
    print(f"Spectral Curve Deviation: {np.mean(spectral):.3f}")
    print(f"PSNR: {np.mean(psnr):.3f}")
    print(f"SSIM: {np.mean(ssim):.3f}")
    print(f"CIEDE2000: {np.mean(ciede):.3f}")
    print(f"LPIPS: {np.mean(lpips_list):.3f}")

    # ---- FID 最后一次性计算 ----
    fid = calculate_fid(imgs1, imgs2)
    print(f"FID: {fid:.3f}")


if __name__ == "__main__":
    # train_folder = r"D:\RGB-T\code-more\ISPRS\dataset\SpA-GAN\SH-TK\0"
    # target_folder = r"D:\RGB-T\code-more\ISPRS\dataset\SH-TK-GT"
    # main(train_folder, target_folder)

    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--train_folder",
        type=str,
        required=True,
        help="训练图像所在文件夹路径"
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        required=True,
        help="GT 图像所在文件夹路径"
    )

    args = parser.parse_args()

    main(args.train_folder, args.target_folder)
