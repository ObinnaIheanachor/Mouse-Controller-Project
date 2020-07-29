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
class Gaze_Estimation:
    '''
    Class for the Gaze_Estimation Model.
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
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=[i for i in self.model.inputs.keys()]
        #self.input_name=next(iter(self.model.inputs))
        print(self.input_name)
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        
        try:
            self.ie = IECore()
            self.ex_net = self.ie.load_network(network=self.model, device_name = self.device)
            return self.ex_net.requests[0].get_perf_counts()

        except Exception as e:
            raise Exception("An error occured while loading the model")

    def predict(self, leftEye, rightEye, headPose):
        
        try:
            
            leftEyeImage=self.preprocess_input(leftEye)
            rightEyeImage=self.preprocess_input(rightEye)
            input_dict = {'head_pose_angles':headPose, 'left_eye_image': leftEyeImage, 'right_eye_image': rightEyeImage}
            print('image dict is ', input_dict)
            infer_request_handle = self.ex_net.start_async(request_id=0, inputs=input_dict)
            infer_status = infer_request_handle.wait()
            start=time.time()
            self.ex_net.infer({'head_pose_angles':headPose, 'left_eye_image': leftEyeImage, 'right_eye_image': rightEyeImage})
            print(f"Time taken to run Gaze_Estimation model on {self.device} is = {time.time()-start} seconds ")
            for i in range(10):
                self.ex_net.infer({'head_pose_angles':headPose, 'left_eye_image': leftEyeImage, 'right_eye_image': rightEyeImage})
                print(f"Time taken to run {i} iterations of Gaze_Estimation model on {self.device} is = {time.time()-start} seconds ")

            endtime = time.time()-start
            if infer_status == 0:
                res = infer_request_handle.outputs[self.output_name]

            print('inference output results for Gaze_Estimation model is',res)
            image = self.preprocess_output(res)
            return image
        except Exception as e:
            raise Exception("An error occured while predicting the outputs")

    def check_model(self):
        
        try:
            supported=[]
            not_supported=[]
            supported = self.ie.query_network(network = self.model, device_name = self.device)
            not_supported = [l for l in self.model.layers.keys() if l not in supported]
            if len(not_supported) != 0:
                logger.info("Unsupported layers found: {}".format(not_supported))
        except Exception as e:
            raise Exception

    def preprocess_input(self, image):
        
        try:
            start=time.time()
            print(self.input_shape)
            dsize=(self.input_shape[3], self.input_shape[2])
            image = cv2.resize(image,(dsize))
            image = image.transpose((2,0,1))
            image = image.reshape(1,*image.shape)
            print(f"Gaze_Estimation model preprocessing time took {time.time()-start} seconds")
            return image
        except Exception as e:
            raise Exception("An error occured while preprocessing the inpouts")


    def preprocess_output(self,coords):
        
        try:
            x = coords[0][0]
            y = coords[0][1]
            return x, y
        except Exception as e:
            raise Exception("An error occured while preprocessing the outputs")
