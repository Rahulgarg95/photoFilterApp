# Importing Required Modules
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from scipy.interpolate import UnivariateSpline
from PIL import Image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def changeBrightness(image, brightness_val):
    # Working with the brightness of image
    brightness = int(((brightness_val + 127) * (127 + 127) / (100 + 100)) - 127)
    #image = cv2.imread(image)
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow

    image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    return image

def changeContrast(image, contrast_val):
    contrast = int(((contrast_val + 64) * (64 + 64) / (100 + 100)) - 64)
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image

def changeSaturation(image, saturation_val):
    value = 1 + (saturation_val / 100)
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvImg[..., 1] = hsvImg[..., 1] * value
    image = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    return image

def changeHue(image, hue_val):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h += hue_val
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def changeVignette(image, vignette_val):
    vignette = int(175.00 / ((vignette_val) / (100.00)))
    rows, cols = image.shape[:2]
    zeros = np.copy(image)
    zeros[:, :, :] = 0
    a = cv2.getGaussianKernel(cols, vignette)
    b = cv2.getGaussianKernel(rows, vignette)
    c = b * a.T
    d = c / c.max()
    zeros[:, :, 0] = image[:, :, 0] * d
    zeros[:, :, 1] = image[:, :, 1] * d
    zeros[:, :, 2] = image[:, :, 2] * d
    return zeros

def changeSharpen(image, sharpen_val):
    amount = int(((sharpen_val - 1) * (3 - 1) / (100 - 0)) + 1)
    kernel_size = (5, 5)
    sigma = 1.0
    threshold = 0
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def filter_cartoon(image):
    img_rgb = image
    num_down = 2
    num_bilateral = 7
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                     blockSize=9, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    return img_cartoon


def filter_edge(image):
    image = cv2.Canny(image, 100, 300)
    return image


def filter_vintage(image):
    rows, cols = image.shape[:2]
    # Create a Gaussian filter
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    filter = 255 * kernel / np.linalg.norm(kernel)
    vintage_im = np.copy(image)
    # for each channel in the input image, we will apply the above filter
    for i in range(3):
        vintage_im[:, :, i] = vintage_im[:, :, i] * filter
    return vintage_im


def filter_blur(image):
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    return image


def filter_grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def filter_monochrome(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

def filter_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def filter_embross(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def filter_warm(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))


def filter_cold(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))


def apply_filter(img, brightnessInput, contrastInput, saturationInput, hueInput, vignetteInput, sharpenInput, effectCheckboxes):
    print('Image Shape: ', img.shape)
    '''if img.shape[1] < 800:
        width = img.shape[1]
    else:
        width = 800

    if img.shape[0] < 800:
        height = img.shape[0]
    else:
        height = 800
    img = cv2.resize(img, (, height))'''
    img = image_resize(img, height = 800)
    print('Resized Image Shape: ', img.shape)

    # Check if Brightness Value is changed
    if brightnessInput != 0:
        img = changeBrightness(img, brightnessInput)

    # Check if contrast value is changed
    if contrastInput != 0:
        img = changeContrast(img, contrastInput)

    # Check if saturation value is changed
    if saturationInput != 0:
        img = changeSaturation(img, saturationInput)

    # Check if hue value is changed
    if hueInput != 0:
        img = changeHue(img, hueInput)

    # Check if vignette value is changed
    if vignetteInput != 0:
        img = changeVignette(img, vignetteInput)

    # Check if sharpen value is changed
    if sharpenInput != 0:
        img = changeSharpen(img, sharpenInput)

    if len(effectCheckboxes) != 0:
        if 'Cartoon' in effectCheckboxes:
            img = filter_cartoon(img)

        if 'Edge' in effectCheckboxes:
            img = filter_edge(img)

        if 'Vintage' in effectCheckboxes:
            img = filter_vintage(img)

        if 'Blur' in effectCheckboxes:
            img = filter_blur(img)

        if 'Grayscale' in effectCheckboxes:
            img = filter_grayscale(img)

        if 'Monochrome' in effectCheckboxes:
            img = filter_monochrome(img)

        if 'Sepia' in effectCheckboxes:
            img = filter_sepia(img)

        if 'Embross' in effectCheckboxes:
            img = filter_embross(img)

        if 'Warm' in effectCheckboxes:
            img = filter_warm(img)

        if 'Cold' in effectCheckboxes:
            img = filter_cold(img)


    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



if __name__ == '__main__':
    #imageInput = gr.inputs.Image(label='Upload an Image',tool='editor',type="numpy",source="webcam")
    imageInput = gr.inputs.Image(label='Upload an Image', tool='editor', type="numpy")
    brightnessInput = gr.inputs.Slider(minimum=-100, maximum=100, default=0, label='Brightness')
    contrastInput = gr.inputs.Slider(minimum=-100, maximum=100, default=0, label='Contrast')
    saturationInput = gr.inputs.Slider(minimum=0, maximum=100, default=0, label='Saturation')
    hueInput = gr.inputs.Slider(minimum=0, maximum=100, default=0, label='Hue')
    vignetteInput = gr.inputs.Slider(minimum=0, maximum=100, default=0, label='Vignette')
    sharpenInput = gr.inputs.Slider(minimum=0, maximum=100, default=0, label='Sharpen')
    effectCheckboxes = gr.inputs.CheckboxGroup(["Cartoon", "Edge", "Vintage", "Blur", "Grayscale", "Monochrome", "Sepia", "Embross", "Warm", "Cold"],
                                               label='Effects')
    imageOutput = gr.outputs.Image(label='Filtered Image')

    desc_str='A custom PhotoFilter App developed to apply multiple filters to a photo.'

    iface = gr.Interface(fn=apply_filter, inputs=[imageInput, brightnessInput, contrastInput, saturationInput, hueInput, vignetteInput,
                                           sharpenInput, effectCheckboxes], outputs=imageOutput,title='PhotoFilter App',
                         description=desc_str ,allow_screenshot=True,live=True,layout='unaligned',verbose=True)
    #iface.launch(share=True)
    iface.launch()