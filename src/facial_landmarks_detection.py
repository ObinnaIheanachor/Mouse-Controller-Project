'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import logging 
import time
import cv2
from openvino.inference_engine import IENetwork, IECore

class Face_landmarks:
    '''
    Class for the Face_landmarks_detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):

        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=0.60
        self.logger = logging.getLogger(__name__)
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        print(self.input_name)
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape        

       

        

    def load_model(self):

  
        try:
            self.ie = IECore()
            self.ex_net = self.ie.load_network(network=self.model, device_name = self.device)
            return self.ex_net.requests[0].get_perf_counts()

        except Exception as e:
            raise Exception("An error occured while loading the model")
        
    def predict(self, image):
        
        try:
            input_img = image
            image=self.preprocess_input(image)
            input_dict = {self.input_name:image}
            print('image dict is ', input_dict)
            infer_request_handle = self.ex_net.start_async(request_id=0, inputs=input_dict)
            infer_status = infer_request_handle.wait()
            start=time.time()
            self.ex_net.infer(input_dict)
            print(f"Time taken to run Face_landmarks_detection model on {self.device} is = {time.time()-start} seconds ")
            for i in range(10):
                self.ex_net.infer(input_dict)
                print(f"Time taken to run {i} iterations of Face_landmarks_detection model on {self.device} is = {time.time()-start} seconds ")

            endtime = time.time()-start
            if infer_status == 0:
                res = infer_request_handle.outputs[self.output_name]

            print('inference output results for Face_landmarks_detection model is',res)
            coords = self.preprocess_output(res)
                       
            return self.denorm_output(coords,input_img)
        except Exception as e:
            raise Exception("There are some problems while predicting pass a review through this function")

    def check_model(self):
        try:
            supported=[]
            not_supported=[]
            supported = self.ie.query_network(network = self.model, device_name = self.device)
            not_supported = [l for l in self.model.layers.keys() if l not in supported]
            if len(not_supported) != 0:
                logger.info("Unsupported layers found: {}".format(not_supported))
        except Exception as e:
            raise Exception("The check method got an error")

    def preprocess_input(self, image):
        
        try:
            start=time.time()
            self.logger.info("the input shape of face model is",self.input_shape)
            self.logger.info("the outshape of  face model is", self.output_shape)
            print(image)
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
            self.logger.info(f"Face_landmarks_detection model preprocessing time took {time.time()-start} seconds")
            return image
        except Exception as e:
            raise Exception
        raise NotImplementedError

    def denorm_output(self, coords,image):
        height = image.shape[0]
        width = image.shape[1]
        
        l_x0 = int(coords[0]*width) - 10
        print(l_x0)
        l_x1 = int(coords[0]*width) + 10
        print(l_x1)
        l_y0 = int(coords[1]* height) - 10
        print(l_y0)
        l_y1 = int(coords[1]*height) + 10
        print(l_y1)
        r_x0 = int(coords[2]*width) - 10
        print(r_x0)
        r_x1 = int(coords[2]*width) + 10
        print(r_x1)
        r_y0 = int(coords[3]*height) - 10
        print(r_y0)
        r_y1 = int(coords[3]*height) + 10
        print(r_y1)
        
        
        l_eye = image[l_y0:l_y1,l_x0:l_x1]
        r_eye = image[r_y0:r_y1,r_x0:r_x1]

        cv2.rectangle(image, (l_x0, l_y0), (l_x1, l_y1), (0, 0, 255), 2)
        cv2.rectangle(image, (r_x0, r_y0), (r_x1, r_y1), (0, 0, 255), 2)

        #cv2.imwrite("FacialLandmark.jpg", image)

        return l_eye,r_eye, image

    
    def preprocess_output(self,coords):

        try:
            
            
            xl=coords[0][0]
            yl=coords[0][1]
            xr=coords[0][2]
            yr=coords[0][3]                

            return (xl,yl,xr,yr)
        except Exception as e:
            raise Exception("an error occured while processing the outputs")
