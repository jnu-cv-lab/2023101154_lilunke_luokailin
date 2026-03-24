import cv2
import numpy as np
import matplotlib.pyplot as plt

# 计算 PSNR 的函数
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 读入图像
img = cv2.imread('zuoye1_tupian.jpg')
if img is None:
    print("错误：找不到图片！请确保 zuoye1_tupian.jpg 在文件夹里")
    exit()

# 显示原图
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转换到 YCrCb 色彩空间
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

print(f"原图尺寸: {Y.shape}")

# 下采样因子
down_factor = 2
h, w = Y.shape

# 下采样
Cb_down = Cb[::down_factor, ::down_factor]
Cr_down = Cr[::down_factor, ::down_factor]

print(f"下采样后 Cb 尺寸: {Cb_down.shape}")

# 用两种插值方法恢复到原尺寸
# 最近邻插值
Cb_upscale_nn = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_NEAREST)
Cr_upscale_nn = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_NEAREST)

# 双线性插值
Cb_upscale_linear = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
Cr_upscale_linear = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# 重建图像
# 最近邻插值重建
ycrcb_nn = cv2.merge([Y, Cr_upscale_nn, Cb_upscale_nn])
img_nn = cv2.cvtColor(ycrcb_nn, cv2.COLOR_YCrCb2BGR)
img_nn_rgb = cv2.cvtColor(img_nn, cv2.COLOR_BGR2RGB)

# 双线性插值重建
ycrcb_linear = cv2.merge([Y, Cr_upscale_linear, Cb_upscale_linear])
img_linear = cv2.cvtColor(ycrcb_linear, cv2.COLOR_YCrCb2BGR)
img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)

#  计算 PSNR
psnr_nn = calculate_psnr(img, img_nn)
psnr_linear = calculate_psnr(img, img_linear)

print(f"\n===== 结果分析 =====")
print(f"下采样因子: {down_factor}")
print(f"最近邻插值 PSNR: {psnr_nn:.2f} dB")
print(f"双线性插值 PSNR: {psnr_linear:.2f} dB")

# 显示结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(Y, cmap='gray')
plt.title('Y Channel')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(Cb_down, cmap='gray')
plt.title(f'Cb Downsampled')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_nn_rgb)
plt.title(f'Nearest Neighbor\nPSNR: {psnr_nn:.2f} dB')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_linear_rgb)
plt.title(f'Bilinear\nPSNR: {psnr_linear:.2f} dB')
plt.axis('off')

# 差值图
diff = np.abs(img.astype(np.float32) - img_linear.astype(np.float32))
diff = (diff / diff.max() * 255).astype(np.uint8)
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
plt.title('Difference Map')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\n结束")