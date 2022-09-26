import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
from numpy import asarray
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

class FID:
    def __init__(self):
        """
        initiates the Inception model for calculating Frechet Inception Distance between two sets of images 
        
        Wikipedia:
        "The Fréchet inception distance (FID) is a metric used to assess the quality of 
        images created by a generative model, like a generative adversarial network (GAN)."

        """
        self.model = self.get_model(input_shape=(299,299,3))

    
    def get_model(self,input_shape,pooling='avg'):
        """ defines the IceptionV3 model for the given input shape of images"""

        return InceptionV3(include_top=False, pooling=pooling, input_shape=input_shape)

    def scale_images(self,images,size):
        """

        returns: scaled images required to pass to the Inception model
                (The input shapes of InceptionV3 model are larger than the image size used in the high energy physics dataset)

        """
        images_list = []
        for image in images:
            new_image = resize(image,size,0)
            images_list.append(new_image)

        return asarray(images_list)
    
    def calculate(self,images_1,images_2):
        """
        calculates the FID score 
        
        Arguments:
          -original set of images
          -generated set of images

        Returns: 
          FID between two sets of images
        
        """
        images_1 = self.scale_images(images_1,(299,299,3))
        images_2 = self.scale_images(images_2,(299,299,3))
        if(images_1.shape != images_2.shape):
            raise Exception('Both set of images must have equal number of images. You passed image data with input shapes {new_images_1.shape} and {new_images_2.shape}')
        images_1 = preprocess_input(images_1)
        images_2 = preprocess_input(images_2)
  
        act1 = self.model.predict(images_1)
        act2 = self.model.predict(images_2)
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        ssdiff = np.sum((mu1-mu2)**2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if iscomplexobj(covmean):
            covmean =covmean.real
  
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean)
        
        return fid


