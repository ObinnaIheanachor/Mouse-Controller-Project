import os
import sys
import logging as log
import time
import cv2
from openvino.inference_engine import IENetwork, IECore
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class FaceDetection:
    '''
    Class for the Face_Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=0.60
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
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            self.ie = IECore()
            self.ex_net = self.ie.load_network(network=self.model, device_name = self.device)
            return self.ex_net.requests[0].get_perf_counts()

        except Exception as e:
            raise Exception("there is a proble encountered while loading the model")

    def predict(self, image,w ,h):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            input_img = image
            image=self.preprocess_input(image)
            input_dict = {self.input_name:image}
            print('image dict is ', input_dict)
            infer_request_handle = self.ex_net.start_async(request_id=0, inputs=input_dict)
            infer_status = infer_request_handle.wait()
            start=time.time()
            self.ex_net.infer(input_dict)
            print(f"Time taken to run Face_Detection model on {self.device} is = {time.time()-start} seconds ")
            for i in range(10):
                self.ex_net.infer(input_dict)
                print(f"Time taken to run {i} iterations of Face_Detection model on {self.device} is = {time.time()-start} seconds ")

            endtime = time.time()-start
            if infer_status == 0:
                res = infer_request_handle.outputs[self.output_name]

            print('inference output results for Face_Detection model is',res)
            image = self.preprocess_output(res, input_img, w, h)
            return image
        except Exception as e:
            raise Exception("there is a probelm encountered in the prediction process")

    def check_model(self):
        try:
            supported=[]
            not_supported=[]
            supported = self.ie.query_network(network = self.model, device_name = self.device)
            not_supported = [l for l in self.model.layers.keys() if l not in supported]
            if len(not_supported) != 0:
                logger.info("Unsupported layers found: {}".format(not_supported))
        except Exception as e:
            raise Exception("some Layers in the model are not supported")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        try:
            start=time.time()
            print("the input shape of face model is",self.input_shape)
            print("the outshape of  face model is", self.output_shape)
            print(image)
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
            print(f"face model preprocessing time took {time.time()-start} seconds")
            return image
        except Exception as e:
            raise Exception("The model preprocessing has some errors")

    def preprocess_output(self,coords,outputs,w, h):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            det = []
            image= ''
            print()
            print('the coords are', coords)
            for obj in coords[0][0]:
              #  print()
                #print('the points obj are',obj[7])
           #     # Draw bounding box for object when it's probability is more than the specified threshold
                if obj[2] > 0.6:
                    #pointer += 1
                    xmin = int(obj[3] * w)
           #         print('xmin', xmin)
                    ymin = int(obj[4] * h)
           #         print('ymin', ymin)
                    xmax = int(obj[5] * w)
           #         print('xmax', xmax)
                    ymax = int(obj[6] * h)
                    p1 = (int(obj[3] * w), int(obj[4] * h))
                    p2 = (int(obj[5] * w), int(obj[6] * h))
                    cv2.rectangle(outputs, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
           #         inf_time_message = "Person detected at accuarcy of {}".format(obj[2]*100)
           #         det.append(obj)
                    image = outputs[ymin:ymax,xmin:xmax]  
            return image
        except Exception as e:
            raise Exception("The model ouput processing got some errors")
