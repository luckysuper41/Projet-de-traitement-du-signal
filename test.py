import demosaicking 
import utils

def main():
    filename = 'kodak19.png'
#    filename = 'goldhill.bmp'
    orimg = utils.getimg(filename)
    
    # Mosaic image
    pattern = [[2, 1], [1, 0]]
    mosaiced_img = utils.mosaicing(orimg, pattern)
    utils.array2img(mosaiced_img).save('imgdb/mosaiced_img.png')
    
    # Demosaic by Hamilton-Adams method
    ha_demosaic = demosaicking.hamilton_demosaicing(mosaiced_img, pattern)
    utils.array2img(ha_demosaic).save('imgdb/ha_demosaiced_img.png')
    Ha_PSNR = utils.PSNR(orimg, ha_demosaic, pattern)
    
    # Demosaic by LMMSE method
    lmmse_demosaic = demosaicking.LMMSE_demosaicing(mosaiced_img, pattern)
    utils.array2img(lmmse_demosaic).save('imgdb/lmmse_demosaiced_img.png')
    LMMSE_PSNR = utils.PSNR(orimg, lmmse_demosaic, pattern)
    
    # Print the PSNR results into a table
    print("Method         #  Red     #  Green    #  Blue     #")
    print("Hamilton-Adams #  %.4f #  %.4f  #  %.4f  #" % Ha_PSNR)
    print("LMMSE          #  %.4f #  %.4f  #  %.4f  #."% LMMSE_PSNR)
          

if __name__ == "__main__":
    main()
    
