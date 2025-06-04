import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
import os
import argparse


def calculate_metrics(corrected, reference):
    """Calculate PSNR and SSIM metrics"""
    # Make sure the image is a floating-point type and normalize to [0, 1].
    if corrected.dtype != np.float32:
        corrected = corrected.astype(np.float32) / 255.0
    if reference.dtype != np.float32:
        reference = reference.astype(np.float32) / 255.0

    # Calculate the window size dynamically (keep it odd)
    def get_valid_win_size(img_shape, base_win=11):
        h, w = img_shape[:2]
        min_dim = min(h, w)
        win_size = min(base_win, min_dim)
        return win_size if win_size % 2 == 1 else win_size - 1

    # Calculate PSNR
    psnr = compare_psnr(reference, corrected, data_range=1.0)

    # Calculate SSIM (auto-fit window size)
    ssim_win = get_valid_win_size(corrected.shape)
    ssim = compare_ssim(
        reference,
        corrected,
        win_size=ssim_win,
        data_range=1.0,
        channel_axis=-1
    )

    return psnr, ssim


def main():
    ap = argparse.ArgumentParser(description='Calculate PSNR and SSIM metrics for image restoration')
    ap.add_argument("-to", "--test_original", required=True, help="path to original testing images")
    ap.add_argument("-td", "--test_restored", required=True, help="path to restored testing images")
    args = vars(ap.parse_args())

    reference_dir = args["test_original"]
    result_dir = args["test_restored"]


    reference_files = set(os.listdir(reference_dir))
    result_files = set(os.listdir(result_dir))
    valid_files = reference_files & result_files

    if not valid_files:
        print("Error: There are no common files in the two folders！！")
        return

    total_psnr, total_ssim = 0.0, 0.0
    valid_count = 0


    metrics_file = os.path.join(result_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Evaluation results (PSNR/SSIM)\n")
        f.write("=" * 50 + "\n")

        for image_name in valid_files:
            try:
                ref_path = os.path.join(reference_dir, image_name)
                res_path = os.path.join(result_dir, image_name)
                reference = io.imread(ref_path)
                corrected = io.imread(res_path)

                if reference.shape != corrected.shape:
                    print(f"Skip {image_name}: Image size mismatch！ (gt: {reference.shape}, restored: {corrected.shape})")
                    continue

                psnr, ssim = calculate_metrics(corrected, reference)
                total_psnr += psnr
                total_ssim += ssim
                valid_count += 1

                result_line = f"{image_name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}\n"
                print(result_line, end='')
                f.write(result_line)

            except Exception as e:
                print(f"Error processing {image_name} : {str(e)}")

        if valid_count > 0:
            avg_psnr = total_psnr / valid_count
            avg_ssim = total_ssim / valid_count
            summary = f"\nAverage: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f} ({valid_count} images in total)"
            print(summary)
            f.write(summary)
        else:
            print("Error: There are no valid image pairs to evaluate")

    print(f"\nEvaluation results have been saved to: {metrics_file}")


if __name__ == '__main__':
    main()