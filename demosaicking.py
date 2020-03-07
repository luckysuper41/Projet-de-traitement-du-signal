import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import convolve
import utils 


def hamilton_demosaicing(mosaiced_img, pattern):
    """ Demosaicing by Hamilton-Adams demosaicing scheme. Missing green components is
    estimated by the directional gradient to avoid the interpolation across the strong 
    direction. 
    :params mosaiced_img: image to be demosaicked.
    :params pattern: corresponding CFA pattern of the mosaiced img.
    
    :return imout: output image after demosaicking.
    """
    imout, mask  = utils.keep_measures(mosaiced_img, pattern)
    # Assign filter for computing forizontal gradient and laplacian
    gfilt = np.zeros((3, 3))
    gfilt[1,:] = np.array([-1, 0, 1])
    lfilt = np.zeros((5,5))
    lfilt[2,:] = np.array([-1, 0, 2, 0, -1])
    # Compute combined gradients
    lx = convolve(mosaiced_img, lfilt)
    ly = convolve(mosaiced_img, lfilt.T)
    dx = np.abs(lx) + np.abs(convolve(mosaiced_img, gfilt))
    dy = np.abs(ly) + np.abs(convolve(mosaiced_img, gfilt.T))
    del gfilt, lfilt
    
    # Compute horizontal and vertical linear interpolations
    hinterp = convolve(mosaiced_img, np.array([[0,0,0],[0.5, 0, 0.5],[0,0,0]]))
    vinterp = convolve(mosaiced_img, np.array([[0,0.5,0], [0,0,0], [0,0.5,0]]))
    
    # Fill out missing green values using directional-based approach
    imout[:,:,1] = imout[:,:,1] + (1 - mask[:,:, 1])*((dx > dy)*( vinterp + ly/4)+ \
              (dx < dy)*(hinterp + lx/4) + (dx == dy)*(hinterp/2+vinterp/2+(lx+ly)/8))
    # Now interpolate red channel and blue channel using estimated green channel
    for plane_num in [0, 2]:
        plane = - mask[:,:, plane_num]*imout[:,:,1] + imout[:,:,plane_num]
        index_x, index_y, data = utils.data_indices(plane, pattern, plane_num)
        interpolator = RectBivariateSpline(index_y, index_x, data)
        imout[:,:,plane_num] = imout[:,:,1] + \
        interpolator(np.arange(0,plane.shape[0],1), np.arange(0,plane.shape[1],1))
        
    return imout
    
def LMMSE_demosaicing(mosaiced_img, pattern, ksize=4, sigma=2):
    """ Linear Minimum Mean Square Error demosaicing
    Perform L.Zhang's demosaicing algorithm. The high quality of green channel will
    be exploited to recover missing data. For missing green values, at red and blue positions,
    we estimate the Primary Difference Signal G - R(B) from two directional interpolations.
    Then the LMMSE will be applied to find out the optimal direction from these two 
    observations. Finally the green channel and red, blue channels will be reconstructed,
    respectively.
    :params mosaiced_img: 2D array, mosaiced image
    :params pattern: 2D list, Bayer pattern
    : return imout: 3D array, output image
    """
    imout, mask = utils.keep_measures(mosaiced_img, pattern)
    # Compute horizontal and vertical estimations of G - R and G - B
    GR, dGR = imout[:,:,0] + imout[:,:,1], imout[:,:,1] - imout[:,:,0]
    GB, dGB = imout[:,:,2] + imout[:,:,1], imout[:,:,1] - imout[:,:,2]
    
    filt = np.array([[0,0,0,0,0],[0,0,0,0,0],[-0.25,0.5,0.5,0.5,-0.25],[0,0,0,0,0],[0,0,0,0,0]])
    dGR_h = dGR + (mask[:,:,0] - mask[:,:,1])*convolve(GR, filt)
    dGR_v = dGR + (mask[:,:,0] - mask[:,:,1])*convolve(GR, filt.T)
    dGB_h = dGB + (mask[:,:,2] - mask[:,:,1])*convolve(GB, filt)
    dGB_v = dGB + (mask[:,:,2] - mask[:,:,1])*convolve(GB, filt.T)
    
    del GR, dGR, GB, dGB, filt
    # Lowpass filtering
    lfilt = np.zeros((2*ksize+1, 2*ksize+1))
    lfilt[ksize,:] = np.exp(-np.arange(-ksize, ksize+1, 1)**2/(2*sigma**2))/((2*np.pi)**0.5*sigma)
    mRG, mBG = mask[:,:,0] + mask[:,:,1], mask[:,:,2] + mask[:,:,1]
    dGR_hs = mRG*convolve(dGR_h, lfilt)
    dGR_vs = mRG*convolve(dGR_v, lfilt.T)
    dGB_hs = mBG*convolve(dGB_h, lfilt)
    dGB_vs = mBG*convolve(dGB_v, lfilt.T)
    
    del lfilt
    
    # Mean filtering
    mfilt = np.zeros((2*ksize+1, 2*ksize+1))
    mfilt[ksize,:] = np.ones((1, 2*ksize+1))/(2*ksize+1)
    # Mean of lowpass filtered observations at R and B positions
    mux_RGhs = convolve(dGR_hs, mfilt)
    mux_RGvs = convolve(dGR_vs, mfilt.T)
    mux_BGhs = convolve(dGB_hs, mfilt)
    mux_BGvs = convolve(dGB_vs, mfilt.T)
    
    # Variance of lowpass filtered noise-free G at R and B positions
    varx_RGhs = convolve(dGR_hs**2, mfilt) - mux_RGhs**2
    varx_RGvs = convolve(dGR_vs**2, mfilt.T) - mux_RGvs**2
    varx_BGhs = convolve(dGB_hs**2, mfilt) - mux_BGhs**2
    varx_BGvs = convolve(dGB_vs**2, mfilt.T) - mux_BGvs**2
    
    # Variance of interpolation error at R and B positions
    vare_RGhs = convolve((dGR_h - dGR_hs)**2, mfilt)
    vare_RGvs = convolve((dGR_v - dGR_vs)**2, mfilt.T)
    vare_BGhs = convolve((dGB_h - dGB_hs)**2, mfilt)
    vare_BGvs = convolve((dGB_v - dGB_vs)**2, mfilt.T)
    
    # LMMSE estimation
    dGR_hhat = (mux_RGhs + varx_RGhs/(varx_RGhs + vare_RGhs)*(dGR_h - mux_RGhs))
    dGR_vhat = (mux_RGvs + varx_RGvs/(varx_RGvs + vare_RGvs)*(dGR_v - mux_RGvs))
    dGB_hhat = (mux_BGhs + varx_BGhs/(varx_BGhs + vare_BGhs)*(dGB_h - mux_BGhs))
    dGB_vhat = (mux_BGvs + varx_BGvs/(varx_BGvs + vare_BGvs)*(dGB_v - mux_BGvs))
    
    # Variance of estimation errors
    varest_RGhs = convolve((dGR_hs - dGR_hhat)**2, mfilt)
    varest_RGvs = convolve((dGR_vs - dGR_vhat)**2, mfilt.T)
    varest_BGhs = convolve((dGB_hs - dGB_hhat)**2, mfilt)
    varest_BGvs = convolve((dGB_vs - dGB_vhat)**2, mfilt.T)
    
    del dGR_hs, dGR_vs, dGB_hs, dGB_vs, mux_RGhs, mux_RGvs, mux_BGhs, mux_BGvs
    # Optimal fusion 
    wh_RG = varest_RGvs/(varest_RGhs + varest_RGvs)
    wh_BG = varest_BGvs/(varest_BGhs + varest_BGvs)
    
    dGR = (wh_RG*dGR_hhat + (1 - wh_RG)*dGR_vhat)*mask[:,:,0]
    dGB = (wh_BG*dGB_hhat + (1 - wh_BG)*dGB_vhat)*mask[:,:,2]
    
    
    imout[:,:,1] = imout[:,:,1] + dGR + imout[:,:,0] + dGB + imout[:,:,2]
    
    ########## Now estimate R and B from G ####################################
    # Interpolate R(B) at B(R) positions
    filt = np.array([[0.25, 0, 0.25],[0,0,0],[0.25, 0, 0.25]])
    imout[:,:,0] = imout[:,:,0] + mask[:,:,2]*(imout[:,:,1] - convolve(dGR, filt))
    imout[:,:,2] = imout[:,:,2] + mask[:,:,0]*(imout[:,:,1] - convolve(dGB, filt))
    
    # Interpolate R and B at G positions
    filt = np.array([[0, 0.25, 0],[0.25, 0, 0.25],[0, 0.25, 0]])
    imout[:,:,0] += mask[:,:,1]*(imout[:,:,1] - convolve(imout[:,:,1]-imout[:,:,0], filt))
    
    imout[:,:,2] += mask[:,:,1]*(imout[:,:,1] - convolve(imout[:,:,1]-imout[:,:,2], filt))
    
    return imout
    



