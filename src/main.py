import cv2
import numpy as np
import argparse
import logging
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetection
from head_pose_estimation import headPoseEstimation
from facial_landmarks_detection import Face_landmarks
from gaze_estimation import Gaze_Estimation
import time
import os
from performanceVisualization import visualize

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    # -- Add required and optional groups
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- create the arguments

    required.add_argument("-m_f", help="path to face detection model", required=True)
    required.add_argument("-m_l", help="path to facial landmarks detection model", required=True)
    required.add_argument("-m_h", help="path to head pose estimation detection model", required=True)
    required.add_argument("-m_g", help="path to gaze detection model", required=True)
    
    optional.add_argument("-d", help="Specify the target device type", default='CPU')
    required.add_argument("-i", help="path to video/image file or 'cam' for webcam", required=True)
    optional.add_argument("-p", help="path to store performance stats", required=False)

    required.add_argument("-o_p", help="output path to all model state folders", required=True)
    optional.add_argument("-vf", help="specify flags from m_f, m_l, m_h, m_g e.g. -vf m_f m_l m_h m_g (seperate each flag by space) for visualization of the output of intermediate models", nargs='+', default=[], required=False)
    optional.add_argument("-modelAr", help="view the model architecture", required=False)

    args = parser.parse_args()

    return args

def main(args):
    # enable logging for the function
    logger = logging.getLogger(__name__)
    
    # grab the parsed parameters
    faceModel=args.m_f
    facial_LandmarksModel=args.m_l
    headPoseEstimationModel=args.m_h
    GazeEstimationModel=args.m_g
    device=args.d
    inputFile=args.i
    output_path =args.o_p
    modelArchitecture = args.modelAr
    visualization_flag = args.vf

        # initialize feed
    single_image_format = ['jpg','tif','png','jpeg', 'bmp']
    if inputFile.split(".")[-1].lower() in single_image_format:
        feed=InputFeeder('image',inputFile)
    elif args.i == 'cam':
        feed=InputFeeder('cam')
    else:
        feed = InputFeeder('video',inputFile)


    ##Load model time face detection
    faceStart_model_load_time=time.time()
    faceDetection = FaceDetection(faceModel, device)
    faceModelView = faceDetection.load_model()
    faceDetection.check_model()
    total_facemodel_load_time = time.time() - faceStart_model_load_time

    ##Load model time headpose estimatiom
    heaadposeStart_model_load_time=time.time()
    headPose = headPoseEstimation(headPoseEstimationModel, device)
    headPoseModelView = headPose.load_model()
    headPose.check_model()
    heaadposeTotal_model_load_time = time.time() - heaadposeStart_model_load_time

    ##Load model time face_landmarks estimation
    face_landmarksStart_model_load_time=time.time()
    face_landmarks = Face_landmarks(facial_LandmarksModel, device)
    faceLandmarksModelView = face_landmarks.load_model()
    face_landmarks.check_model()
    face_landmarksTotal_model_load_time = time.time() - face_landmarksStart_model_load_time

    ##Load model time face_landmarks estimation
    GazeEstimationStart_model_load_time=time.time()
    GazeEstimation = Gaze_Estimation(GazeEstimationModel, device)
    GazeModelView = GazeEstimation.load_model()
    GazeEstimation.check_model()
    GazeEstimationTotal_model_load_time = time.time() - GazeEstimationStart_model_load_time

    if modelArchitecture == 'yes' :
        print("The model architecture of gaze mode is ", GazeModelView)
        print("model architecture for landmarks is" , faceLandmarksModelView)
        print("model architecture for headpose is" , headPoseModelView )
        print("model architecture for face is" , faceModelView )

        # count the number of frames
    frameCount = 0
    input_feeder = InputFeeder('video', inputFile)
    w,h = feed.load_data() 
    for _, frame in feed.next_batch():

        if not _:
            break
        frameCount+=1
        key = cv2.waitKey(60)
        start_imageface_inference_time = time.time()
        imageface = faceDetection.predict(frame,w,h)
        imageface_inference_time = time.time() -  start_imageface_inference_time

        if 'm_f' in visualization_flag:
            cv2.imshow('cropped face', imageface)
            
        if type(imageface)==int:
            logger.info("no face detected")
            if key==27:
                break
            continue

        start_imagePose_inference_time = time.time()
        imageAngles, imagePose = headPose.predict(imageface)
        imagePose_inference_time = time.time() -  start_imagePose_inference_time

        if 'm_h' in visualization_flag:
            cv2.imshow('Head Pose Angles', imagePose)

        start_landmarkImage_inference_time = time.time()
        leftEye, rightEye, landmarkImage = face_landmarks.predict(imageface)
        landmarkImage_inference_time = time.time() -  start_landmarkImage_inference_time

        if leftEye.any() == None or rightEye.any() == None:
            logger.info("image probably too dark or eyes covered, hence could not detect landmarks")
            continue

        if 'm_l' in visualization_flag:
            cv2.imshow('Face output', landmarkImage)

        start_GazeEstimation_inference_time = time.time()
        x , y = GazeEstimation.predict(leftEye, rightEye, imageAngles)
        GazeEstimation_inference_time = time.time() -  start_GazeEstimation_inference_time

        if 'm_g' in visualization_flag:
#             cv2.putText(landmarkedFace, "Estimated x:{:.2f} | Estimated y:{:.2f}".format(x,y), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0),1)
            cv2.imshow('Gaze Estimation', landmarkImage)

        mouseVector=MouseController('medium','fast')

        if frameCount%5==0:
            mouseVector.move(x,y)

        if key==27:
            break

        if imageface_inference_time != 0 and landmarkImage_inference_time != 0 and imagePose_inference_time != 0 and GazeEstimation_inference_time != 0:

            fps_face = 1/imageface_inference_time
            fps_landmark = 1/landmarkImage_inference_time
            fps_headpose = 1/imagePose_inference_time
            fps_gaze = 1/GazeEstimation_inference_time
            
            
            with open(os.path.join(output_path ,device , 'face', 'face_stats.txt'), 'w') as f:
                f.write(str(imageface_inference_time)+'\n')
                f.write(str(fps_face)+'\n')
                f.write(str(total_facemodel_load_time)+'\n')

            
            with open(os.path.join(output_path, device , 'landmark', 'landmark_stats.txt'), 'w') as f:
                f.write(str(landmarkImage_inference_time)+'\n')
                f.write(str(fps_landmark)+'\n')
                f.write(str(face_landmarksTotal_model_load_time)+'\n')

            
            with open(os.path.join(output_path, device,'headpose', 'headpose_stats.txt'), 'w') as f:
                f.write(str(imagePose_inference_time)+'\n')
                f.write(str(fps_headpose)+'\n')
                f.write(str(heaadposeTotal_model_load_time)+'\n')

            
            with open(os.path.join(output_path, device,'gaze', 'gaze_stats.txt'), 'w') as f:
                f.write(str(GazeEstimation_inference_time)+'\n')
                f.write(str(fps_gaze)+'\n')
                f.write(str(GazeEstimationTotal_model_load_time)+'\n')

        
    
    logger.info("The End")
    VIS = visualize(output_path, device)
    VIS.visualize1()
    VIS.visualize2()
    VIS.visualize3()
    cv2.destroyAllWindows()
    feed.close()


if __name__ == '__main__':
    args=get_args()
    main(args)
        
