from skimage import img_as_ubyte
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class AverageMeter():
	def __init__(self,):
		self.reset()

	def reset(self,):
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.count += n
		self.sum += val * n
		self.avg = self.sum / self.count

def batch_PSNR(img, imclean):
    PSNR = 0.
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img, Iclean = img_as_ubyte(Img), img_as_ubyte(Iclean)
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i], Img[i], data_range=255)
    return (PSNR / Img.shape[0])

def batch_MSE(img, imclean):
    MSE = 0.
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img, Iclean = img_as_ubyte(Img), img_as_ubyte(Iclean)
    for i in range(Img.shape[0]):
        MSE += compare_mse(Iclean[i], Img[i])
    return (MSE / Img.shape[0])

def batch_SSIM(img, imclean):
    SSIM = 0.
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img, Iclean = img_as_ubyte(Img), img_as_ubyte(Iclean)
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Img[i].transpose((1,2,0)),
                             Iclean[i].transpose((1,2,0)),
                             data_range=255, gaussian_weights=True, 
                             channel_axis=2, use_sample_covariance=False)
    return (SSIM / Img.shape[0])