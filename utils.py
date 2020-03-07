import numpy as np
from PIL import Image 

def mosaicing(img, pattern):
    """ Modeling of the mosaicing process. 
    :params img: RGB image, input image to be mosaicked.
    :params pattern: Bayer's color filter array (Bayer CFA).
    :return imout: output image after the mosaic process.
    Each CFA is characterized
    by an "array" of 4 components [[a, b], [c, d]] where the first two characters correspond
    to the first row of the given pattern.
    By convention, we denote 0 for red, 1 for green and 2 for blue channels. For example, 
    if pattern is [[0, 1], [1, 2]] we have RGGB.  
    RGGB. 
    
    """
    imout = np.zeros((img.shape[0], img.shape[1]))
    for i in range(2):
        for j in range(2):
            imout[i::2, j::2] = img[i::2, j::2, pattern[i][j]]
        
    return imout

def data_indices(img, pattern, channel):
    """ 1D arrays of coordinates indicating measurements position and its corresponding
    data.
    :params img: 2D array image
    :params pattern: 2D list, Bayer pattern
    :params channel: integer, denoting the desire channel
    
    :returns index_x: 1D array, horizontal coordinates
    :returns index_y: 1D array, vertical coordinates
    :returns data: 2D array, available measures
    """
    h, w = img.shape
    for i in range(2):
        for j in range(2):
            if pattern[i][j] == channel:
                index_y = np.arange(i, h, 2)
                index_x = np.arange(j, w, 2)
                data =  img[i::2, j::2]
    
    return index_x, index_y, data
    
def keep_measures(mosaiced_img, pattern):
    """ Copy the measured pixels in the mosaiced image into a new RGB image
    :params img: 2D array, mosaiced image
    :params pattern: 2D list, Bayer pattern
    
    :return imout: 3D array
    """
    h, w = mosaiced_img.shape
    imout = np.zeros((h, w, 3))
    mask = np.zeros((h, w, 3))
    for i in range(2):
        for j in range(2):
            imout[i::2, j::2, pattern[i][j]] = mosaiced_img[i::2,j::2]
            mask[i::2, j::2, pattern[i][j]] = 1
    
    return imout, mask

def PSNR(orimg, estimg, pattern):
    """ Compute PSNR metrics for a certain demosaicking method. This metrics is computed
    for every channel, defining by the CFA pattern.
    :params orimg: 3D array, original image
    :params estimg: 3D array, estimation image
    :params pattern: 2D list, Bayer CFA pattern
    
    :returns PSNR: tuple contains PSNR of R, G, B channels
    
    """
    PSNR = [0]*3
    _, mask = keep_measures(orimg[:, :, 0], pattern)
    for i in range(3):
        diff = orimg[:,:,i] - estimg[:,:,i]
        PSNR[i] = 10*np.log10(255**2/(np.linalg.norm((1-mask[:,:,i])*diff)**2/(1-mask[:,:,i]).sum()))
        
    return tuple(PSNR)
    
def getimg(filename):
    """ Get image array given filename.
    :params filename: string, filename
    :return img: numpy array representing image data (either 2D or 3D)
    """
    return np.asarray(Image.open('imgdb/'+filename))

def array2img(array):
    """ Convert array to Image object.
    :params array: numpy array.
    :return Img: Image object.
    """
    if len(array.shape) == 2:
        return Image.fromarray(np.clip(array, 0, 255).astype('uint8'), mode='L')
    elif len(array.shape) == 3:
        return Image.fromarray(np.clip(array, 0, 255).astype('uint8'), mode='RGB')
    else:
        print('Income array is not at appropriate shape!')


    
    