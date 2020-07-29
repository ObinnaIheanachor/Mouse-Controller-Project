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

class headPoseEstimation:
    '''
    Class for the head_pose_estimation Model.
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


    def predict(self, pimage):
        
        try:
            input_img = pimage
            print(pimage)
            image=self.preprocess_input(pimage)
            input_dict = {self.input_name:image}
            self.logger.info('image dict is ', input_dict)
            infer_request_handle = self.ex_net.start_async(request_id=0, inputs=input_dict)
            infer_status = infer_request_handle.wait()
            start=time.time()
            self.ex_net.infer(input_dict)
            self.logger.info(f"Time taken to run headpose model on {self.device} is = {time.time()-start} seconds ")
            for i in range(10):
                self.ex_net.infer(input_dict)
                print(f"Time taken to run {i} iterations of headpose model on {self.device} is = {time.time()-start} seconds ")

            endtime = time.time()-start
            if infer_status == 0:
                res = infer_request_handle.outputs[self.output_name]

            print('inference output results for headpose model is',res)

            return self.preprocess_output(res, pimage)
        except Exception as e:
            raise Exception("An error occured while predicting the output")

    def check_model(self):
        try:
            supported=[]
            not_supported=[]
            supported = self.ie.query_network(network = self.model, device_name = self.device)
            not_supported = [l for l in self.model.layers.keys() if l not in supported]
            if len(not_supported) != 0:
                logger.info("Unsupported layers fountd: {}".format(not_supported))
        except Exception as e:
            raise Exception("An error occured while checking the model")


    def preprocess_input(self, image):

        try:
            start=time.time()
            self.logger.info("the input shape of headpose model is",self.input_shape)
            self.logger.info("the outshape of  headpose model is", self.output_shape)
            print(image)
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
            self.logger.info(f"headpose model preprocessing time took {time.time()-start} seconds")
            return image
        except Exception as e:
            raise Exception("An error occured while processing the inputs")


    def preprocess_output(self,coords, image):

        try:
            angles = []
        
            angles.append(coords[0][0])
            angles.append(coords[0][0])
            angles.append(coords[0][0])
            
            cv2.putText(image, "Estimated yaw:{:.2f} | Estimated pitch:{:.2f}".format(angles[0],angles[1]), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0),1)
            print()
            cv2.putText(image, "Estimated roll:{:.2f}".format(angles[2]), (10,70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0),1)
            return angles, image
        except Exception as e:
            raise Exception("An error occured while processing the outputs")
