import matplotlib.pyplot as plt
import logging
import threading
import time
import os 

class visualize:
    def __init__(self,outputPath, device):

        self.model = ['face', 'gaze', 'headpose', 'landmark']
        self.device_list=['cpu', 'gpu', 'fpga', 'vpu']
        self.stat_list=['face_stats.txt', 'gaze_stats.txt', 'headpose_stats.txt', 'landmark_stats.txt']
        self.inference_time=[]
        self.model_load_time=[]
        self.fps_time=[]
        self.outputPath = outputPath
        self.device = device
        i = 0
        
        for stat in self.stat_list:

            with open(os.path.join(outputPath, device , self.model[i], stat ), 'r') as f:
                    self.inference_time.append(float(f.readline().split("\n")[0]))
                    self.fps_time.append(float(f.readline().split("\n")[0]))
                    self.model_load_time.append(float(f.readline().split("\n")[0]))
            i+=1



    def visualize1(self):
        plt.bar(self.model, self.inference_time)
        plt.xlabel(self.device + "used")
        plt.ylabel("Total Inference Time in Seconds")
        plt.show()

    def visualize2(self):
        plt.bar(self.model, self.model_load_time)
        plt.xlabel(self.device + "used")
        plt.ylabel("Model Loading Time in Seconds")
        plt.show()

    def visualize3(self):
        plt.bar(self.model, self.fps_time)
        plt.xlabel(self.device + "used")
        plt.ylabel("Frames Per Seconds")
        plt.show()        



    
