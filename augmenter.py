import numpy as np
import cv2

class Augmenter:
    
    def generate_random_lines(imshape,slant,drop_length):
        drops=[]
        streaks = np.random.randint(300, 500)
        for i in range(streaks):
            if slant<0:
                x= np.random.randint(slant,imshape[1])
            else:
                x= np.random.randint(0,imshape[1]-slant)
            y= np.random.randint(0,imshape[0]-drop_length)
            drops.append((x,y))
        return drops
    
    def __call__(self, image):
        imshape = image.shape
        slant_extreme = np.random.randint(5, 10)
        slant= np.random.randint(-slant_extreme,slant_extreme)
        drop_length = np.random.randint(12, 21)
        drop_width = 2
        drop_color = (200, 200, 200) 
        rain_drops= generate_random_lines(imshape, slant, drop_length)
        for rain_drop in rain_drops:
            cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
        blur = np.random.randint(3, 8)
        image= cv2.blur(image, (blur, blur)) 
        brightness_coefficient = 0.7 
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient 
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
        return image_RGB