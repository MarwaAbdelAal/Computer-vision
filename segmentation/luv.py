import numpy as np
import cv2
import matplotlib.pyplot as plt

def RGB2LUV(image):
    copied_image = image.copy()
    width, height = copied_image.shape[:2]
    copied_image =copied_image /255.0

    for i in range(width):
        
        for j in range(height):
            
            x = 0.412453 * copied_image[i, j][0] + 0.357580 * copied_image[i, j][1] + 0.180423 * copied_image[i, j][2]
            y = 0.212671 * copied_image[i, j][0] + 0.715160 * copied_image[i, j][1] + 0.072169 * copied_image[i, j][2]
            z = 0.019334 * copied_image[i, j][0] + 0.119193 * copied_image[i, j][1] + 0.950227 * copied_image[i, j][2]

            if (y > 0.008856):
                L = (116.0 * (y **(1/3)) ) - 16.0
            else:
                L = 903.3 * y
            
            u_dash = 4.0*x /( x + (15.0*y ) + 3.0*z) 
            v_dash = 9.0*y /( x + (15.0*y ) + 3.0*z) 

            U = 13.0 * L * (u_dash -0.19793943)
            V = 13.0 * L * (v_dash -0.46831096)

            image [i,j] [0] = ( 255.0/100.0) *L
            image [i,j] [1] = ( 255.0/ 354.0) *(U+134.0 )
            image [i,j] [2] = (255.0/ 262.0) *(V +140.0) 

    print (image [i,j] [0] )
    print( image [i,j] [1])
    print( image [i,j] [2])

    return image.astype(np.uint8)

if __name__ == "__main__":
        
    img = cv2.imread("images/landscape.png")
    luv_img_cv = cv2.cvtColor(img,cv2.COLOR_BGR2Luv)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    luv_img = RGB2LUV(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].imshow(img_rgb)
    ax[0].set_title('Original Image')
    ax[0].set_axis_off()
    
    print (luv_img[0,0])
    ax[1].imshow(luv_img )
    ax[1].set_title('Luv Image')
    ax[1].set_axis_off()
    print (luv_img_cv[0,0])
    
    ax[2].imshow(luv_img_cv  )
    ax[2].set_title('Luv Image cv')
    ax[2].set_axis_off()
    plt.tight_layout()
    plt.show()