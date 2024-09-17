# IMPLEMENTATION OF A FURTHER ENHANCED ADAPTATION
# OF THE REINHARD COLOUR TRANSFER METHOD.
#
# Python coding by Minh Nguyen Hoang
# https://github.com/minh-nguyenhoang
# Original C++ implementation by Terry Johnson
# https://github.com/TJCoding/Enhanced-Image-Colour-Transfer-2

import cv2
import numpy as np
RGB2LMS = np.array([[0.3811, 0.5783, 0.0402],
                    [0.1967, 0.7244, 0.0782],
                    [0.0241, 0.1288, 0.8444]]).astype(np.float32)
LMS2LAB = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                    [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
                    [1/np.sqrt(2), -1/np.sqrt(2), 0]]).astype(np.float32)

INV_LMS2LAB = np.linalg.inv(LMS2LAB)
INV_RGB2LMS = np.linalg.inv(RGB2LMS)

# Here LAB refers to the L-alpha-beta colour space rather than CIELAB.
def cvt_RGB2LAB(img):
    img = np.maximum(img, 1.)
    img_lms = cv2.transform(img, RGB2LMS)
    img = np.maximum(img, 1.)
    img_lms = np.log(img_lms).astype(np.float32)/np.log(10)
    img_lab = cv2.transform(img_lms, LMS2LAB)

    return img_lab

def cvt_LAB2RGB(img):
    img_lms = cv2.transform(img, INV_LMS2LAB)
    img_lms = np.exp(img_lms).astype(np.float32)
    img_lms = np.power(img_lms, np.log(10.0))
    img_rgb = cv2.transform(img_lms, INV_RGB2LMS)
    return img_rgb

def channel_conditioning(t_channel, s_channel):
    wval = float(0.25)

    _, mask = cv2.threshold(s_channel, 0, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    s_mean_U = np.sum(s_channel*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) ## mean for element above 0
    w_U = np.exp(-s_channel * wval/s_mean_U)
    w_U = (1 - w_U) * (1 - w_U) 
    w_mean = np.sum(w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) ## mean for element above 0
    channel_U = np.power(s_channel, 4)
    s_mean_U = np.sum(channel_U*w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1)/w_mean

    inv_mask = 1 - mask
    s_mean_L = np.sum(s_channel*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0

    w_L = np.exp(-s_channel * wval/s_mean_L)
    w_L = (1 - w_L) * (1 - w_L) 
    w_mean = np.sum(w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0
    channel_L = np.power(s_channel, 4)
    s_mean_L = np.sum(channel_L*w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1)/w_mean


    _, mask = cv2.threshold(t_channel, 0, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    t_mean_U = np.sum(t_channel*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) ## mean for element above 0
    w_U = np.exp(-t_channel * wval/t_mean_U)
    w_U = (1 - w_U) * (1 - w_U) 
    w_mean = np.sum(w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) + 1e-6 ## mean for element above 0
    channel_U = np.power(t_channel, 4)
    t_mean_U = np.sum(channel_U*w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1)/w_mean

    inv_mask = 1 - mask
    t_mean_L = np.sum(t_channel*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0
    w_L = np.exp(-t_channel * wval/t_mean_L)
    w_L = (1 - w_L) * (1 - w_L) 
    w_mean = np.sum(w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0
    channel_L = np.power(t_channel, 4)
    t_mean_L = np.sum(channel_L*w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1)/w_mean

    k = np.sqrt(np.sqrt(s_mean_U/t_mean_U))
    t_channel_U = (1 + w_U*(k-1))*t_channel

    k = np.sqrt(np.sqrt(s_mean_L/t_mean_L))
    t_channel_L = (1 + w_L*(k-1))*t_channel

    t_chanel_normalized: np.ndarray = t_channel_U * mask + t_channel_L * inv_mask
    t_mean, t_std = t_chanel_normalized.mean((0,1)), t_chanel_normalized.std((0,1))

    t_chanel_normalized = (t_chanel_normalized - t_mean)/t_std

    return t_chanel_normalized


def adjust_covariance(t_lab: np.ndarray, s_lab: np.ndarray, cross_covariance_limit = 0.5):
    t_lab = t_lab.copy()
    if cross_covariance_limit != 0:
        tcrosscorr = np.mean(t_lab[...,1] * t_lab[...,2], axis= (0,1))
        scrosscorr = np.mean(s_lab[...,1] * s_lab[...,2], axis= (0,1))

        W1 = 0.5 * np.sqrt((1 + scrosscorr) / (1 + tcrosscorr)) + \
             0.5 * np.sqrt((1 - scrosscorr) / (1 - tcrosscorr))
        W2 = 0.5 * np.sqrt((1 + scrosscorr) / (1 + tcrosscorr)) - \
             0.5 * np.sqrt((1 - scrosscorr) / (1 - tcrosscorr))
        
        if abs(W2) > cross_covariance_limit * abs(W1):
            W2 = np.copysign(cross_covariance_limit * W1, W2)
            norm = 1.0 / np.sqrt(W1**2 + W2**2 + 2 * W1 * W2 * tcrosscorr)
            W1 *= norm
            W2 *= norm

        z1 = t_lab[...,1].copy()
        t_lab[...,1] = W1 * z1 + W2 * t_lab[...,2]
        t_lab[...,2] = W1 * t_lab[...,2] + W2 * z1

    return t_lab

def core_processing(tgt, src, cross_covariance_limit = 0.5, reshaping_iteration = 1, shader_val = 0.5):
    tgtf = cvt_RGB2LAB(tgt)
    srcf = cvt_RGB2LAB(src)

    tgt_mean, tgt_std = cv2.meanStdDev(tgtf)
    src_mean, src_std = cv2.meanStdDev(srcf)

    tgt_mean, tgt_std = tgt_mean.reshape(-1).astype(np.float32), tgt_std.reshape(-1).astype(np.float32)
    src_mean, src_std = src_mean.reshape(-1).astype(np.float32), src_std.reshape(-1).astype(np.float32)

    t_lab = (tgtf - tgt_mean)/tgt_std
    s_lab = (srcf - src_mean)/src_std

    for j in range(reshaping_iteration,(reshaping_iteration + 1)//2, -1):
        t_lab[...,1:] = channel_conditioning(t_lab[...,1:], s_lab[...,1:])
    

    t_lab = adjust_covariance(t_lab, s_lab, cross_covariance_limit)

    for j in range((reshaping_iteration + 1)//2):
        t_lab[...,1:] = channel_conditioning(t_lab[...,1:], s_lab[...,1:])
    

    src_mean, src_std = src_mean.copy(), src_std.copy()
    src_mean[0] = shader_val*src_mean[0] + (1-shader_val)*tgt_mean[0]
    src_std[0] = shader_val*src_std[0] + (1-shader_val)*tgt_std[0]

    t_lab = t_lab * src_std + src_mean

    res_rgb = cvt_LAB2RGB(t_lab)
    return res_rgb


def adjust_saturation(img, origin_img, saturation_val = -1):
    # Convert images from RGB to HSV color space
    img: np.ndarray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    origin_img: np.ndarray = cv2.cvtColor(origin_img.astype(np.float32), cv2.COLOR_RGB2HSV)
    if saturation_val < 0:
        # Calculate saturation_val as the ratio of the max saturation value
        # in the original image to the max value in the processed image
        amax1 = np.max(img[...,1])
        amax2 = np.max(origin_img[...,1])
        saturation_val = amax2 / amax1

    if saturation_val != 1.:
        # Compute weighted mix of the processed target and original saturation channels
        # origin_img[...,1] = cv2.addWeighted(img[...,1], saturation_val, origin_img[...,1], 1 - saturation_val, 0.0)
        origin_img[...,1] = img[...,1]*(saturation_val) + origin_img[...,1]*(1-saturation_val)

        # Create a mask where the processed image's saturation exceeds the original target image's saturation
        mask = cv2.threshold(img[...,1] - origin_img[...,1], 0, 1, cv2.THRESH_BINARY)[1]
        mask = mask.astype(np.uint8)
 
        # Create the modified reference saturation channel
        origin_img[...,1] = origin_img[...,1] * mask + img[...,1]*(1-mask)

        # Match the mean and standard deviation of the processed image's saturation channel
        # to the modified reference saturation channel
        # tmean, tdev = cv2.meanStdDev(img[...,1])
        # tmpmean, tmpdev = cv2.meanStdDev(origin_img[...,1])

        tmean, tdev = img[...,1].mean((0,1)), img[...,1].std((0,1))
        tmpmean, tmpdev = origin_img[...,1].mean((0,1)), origin_img[...,1].std((0,1))

        tmp_ = img[...,1].copy().astype(np.float32)
        img[...,1] = (tmp_ - tmean) / tdev * tmpdev + tmpmean
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img

def full_shading(img, ori_img, src_img, extra_shading = True, shader_val = 0.5):
    # Matches the grey shade distribution of the modified target image
    # to that of a notional shader image which is a linear combination
    # of the original target and source image as determined by 'shader_val'.

    if extra_shading:
        # Convert images to grayscale
        greyt = np.dot(ori_img[..., :3], [0.2989, 0.5870, 0.1140])
        greyp = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        greys = np.dot(src_img[..., :3], [0.2989, 0.5870, 0.1140])

        # Standardize the grayscale images for the source and target
        smean, sdev = np.mean(greys), np.std(greys)
        tmean, tdev = np.mean(greyt), np.std(greyt)
        greyt = (greyt - tmean) / tdev

        # Rescale the standardized grayscale target image
        greyt = greyt * (shader_val * sdev + (1.0 - shader_val) * tdev) \
                + shader_val * smean + (1.0 - shader_val) * tmean

        # Ensure there are no zero or negative values in the grayscale images
        min_val = 1.
        greyp = np.maximum(greyp, min_val)  # Guard against zero divide
        greyt = np.maximum(greyt, 0.0)      # Guard against negative values

        # Rescale each color channel of the processed image
        img = img / greyp[..., None] * greyt[..., None]

    return img

def final_adjustment(img, ori_img, tint_val = 1., modified_val = 1.):
    # Implements a change to the tint of the final image and
    # its degree of modification if a change is specified.

    # If 100% tint is not specified, compute a weighted average
    # of the processed image and its grayscale representation.
    if tint_val != 1.0:
        # Convert to grayscale
        grey = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])[..., None]

        # Apply the tint by adjusting each channel
        img = tint_val * img + (1.0 - tint_val) * grey
        

    # If 100% image modification is not specified, compute a weighted average
    # of the processed image and the original target image.
    if modified_val != 1.0:
        img = modified_val * img + (1.0 - modified_val) * ori_img

    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Specify the image files that are to be processed,
    # where 'source image' provides the colour scheme
    # that is to be applied to 'target image'.
    # Note that image files are assumed to be in a Folder named 'Images'
    _src = cv2.imread(r'Images\Vase_source.jpg')[..., ::-1].copy()
    _tgt = cv2.imread(r'Images\Vase_target.jpg')[..., ::-1].copy()

    # Select the processing options
    reshaping_iteration = 1
    cross_covariance_limit = .5 # On a scale where 1 denotes 100%
    shader_val = .5             # On a scale where 1 denotes 100%
    saturation_val = -1.
    extra_shading = True
    tint_val = 1.               # On a scale where 1 denotes 100%
    modified_val = 1.           # On a scale where 1 denotes 100%

    # For a full explanation of parameter choices and the associated 
    # methodology see the comments in the original C++ coding at
    # https://github.com/TJCoding/Enhanced-Image-Colour-Transfer-2

    _core = core_processing(_tgt,_src, reshaping_iteration= reshaping_iteration, cross_covariance_limit= cross_covariance_limit, shader_val= shader_val)
    _res = adjust_saturation(_core, _tgt, saturation_val= saturation_val)
    _shaded = full_shading(_res, _tgt, _src, extra_shading= extra_shading, shader_val = shader_val)
    _final = final_adjustment(_shaded, _tgt, tint_val= tint_val, modified_val= modified_val)

    _core   =np.clip(_core, 0, 255).astype(np.uint8)
    _res    =np.clip(_res, 0, 255).astype(np.uint8)
    _shaded =np.clip(_shaded, 0, 255).astype(np.uint8)

    view_row = False

    if view_row:
        fig, ax = plt.subplots(1,6, figsize = (30,5))
        fig.add_artist(plt.Line2D([0.375, 0.375], [0.1,0.9], transform=fig.transFigure, color="black"))
        ax[0].imshow(_tgt)
        ax[0].set_title("Target Image")
        ax[0].axis("off")
        ax[1].imshow(_src)
        ax[1].set_title("Palette Image")
        ax[1].axis("off")
        ax[2].imshow(_core)
        ax[2].set_title("After core processing")
        ax[2].axis("off")
        ax[3].imshow(_res)
        ax[3].set_title("After saturation processing")
        ax[3].axis("off")
        ax[4].imshow(_shaded)
        ax[4].set_title("After full shading")
        ax[4].axis("off")
        ax[5].imshow(_final)
        ax[5].set_title("Final Image")
        ax[5].axis("off")
        # fig.show()
    else:
        fig, ax = plt.subplots(2,3, figsize = (20,10))
        fig.add_artist(plt.Line2D([0.375, 0.375], [0.1,0.9], transform=fig.transFigure, color="black"))
        ax[0,0].imshow(_tgt)
        ax[0,0].set_title("Target Image")
        ax[0,0].axis("off")
        ax[1,0].imshow(_src)
        ax[1,0].set_title("Palette Image")
        ax[1,0].axis("off")
        ax[0,1].imshow(_core)
        ax[0,1].set_title("After core processing")
        ax[0,1].axis("off")
        ax[0,2].imshow(_res)
        ax[0,2].set_title("After saturation processing")
        ax[0,2].axis("off")
        ax[1,1].imshow(_shaded)
        ax[1,1].set_title("After full shading")
        ax[1,1].axis("off")
        ax[1,2].imshow(_final)
        ax[1,2].set_title("Final Image")
        ax[1,2].axis("off")
        # fig.show()

    plt.show()