# Computer Pointer Controller

This project runs multiple models in the same machine and coordinate the flow of data between those models. In this project, we will use a gaze detection model to control the mouse pointer of your computer. We will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.

## Project Set Up and Installation

1. Download and install Intel OpenVino Toolkit 

2. Copy these files in the directories as stored here.

3. Initialize the openvino toolkit by:
```
    "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```
4. Download the required models using the following commands:

1st Face Detection Model:-
```
    python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "face-detection-adas-binary-0001"
```
2nd Facial Landmarks Detection Model:-
```
    python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "landmarks-regression-retail-0009"
```
3rd Head Pose Estimation Model:-
```
    python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "head-pose-estimation-adas-0001"
```     
4th Gaze Estimation Model:-
```    
    python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "gaze-estimation-adas-0002"
```

## Demo

1. Change the directory to src directory of project <br>
```
cd <project-path>/src
```
2. Run the main.py file
Example:<br>
```
python main.py -m_f <Path of xml file of face detection model>-m_l <Path of xml file of facial landmarks detection model>-m_h <Path of xml file of head pose estimation model>-m_g <Path of xml file of gaze estimation model>-i <Path of input video file or enter cam for taking input video from webcam
```
```
python main.py -fd <Path of xml file of face detection model>
-m_f <Path of xml file of face detection model>
-m_l <Path of xml file of facial landmarks detection model>
-m_h <Path of xml file of head pose estimation model>
-m_g <Path of xml file of gaze estimation model>
-i <Path of input video file or enter cam for taking input video from webcam> 
```

## Documentation

Directory Structure of the project

computer pointer controller 
          |-- README.md
          |-- bin
                |-- demo.mp4
          |-- requirements.txt
	  |-- src
                |-- __pycache__
			|-- face_detection.cpython-36
			|-- facial_landmarks_detection.cpython-36
			|-- gaze_estimation.cpython-36
			|-- head_pose_estimation.cpython-36
			|-- input_feeder.cpython-36
			|-- mouse_controller.cpython-36
			|-- performanceVisualization.cpython-36
		|-- output_path
			|-- cpu
				|-- face
                		|-- gaze
                		|-- headpose
				|-- landmark
			|-- fpga
				|-- face
                		|-- gaze
                		|-- headpose
				|-- landmark
			|-- gpu
				|-- face
                		|-- gaze
                		|-- headpose
				|-- landmark
			|-- MYRIAD
				|-- face
                		|-- gaze
                		|-- headpose
				|-- landmark	
                |-- face_detection.py
                |-- facial_landmark_detection.py
                |-- gaze_estimation.py
                |-- head_pose_estimation.py
                |-- input_feeder.py
                |-- main.py
                |-- model.py
                |-- mouse_controller.py

\src folder contains all the source files:-

1. main.py : Users need to run main.py file for running the app.

2. facedetect.py: Contains preprocession of video frame, perform infernce on it and detect the face, postprocess the outputs.
     
3. Landmarkdetect.py: Takes the deteted face as input, preprocessed it, perform inference on it and detect the eye landmarks, postprocess the outputs.
     
4. headposedetect.py: Take the detected face as input, preprocessed it, perform inference on it and detect the head postion.
     
5. gazeestimate.py: Take the left eye, rigt eye, head pose angles as inputs, preprocessed it, perform inference and predict the gaze vector, postprocess the outputs.
     
6. input_feeder.py: Contains InputFeeder class which initialize VideoCapture as per the user argument and return the frames one by one.
     
7. mousecontroller.py: Contains MouseController class.
 
\bin folder contains demo video which user can use for testing the app.

Following are commands line arguments that can use for while running the main.py file ` python main.py `:-

  1. -h     (required) : Information about all the command.
  2. -m_f   (required) : Path of Face Detection model's xml file.
  3. -m_h   (required) : Path of Head Pose Estimation model's xml file.
  4. -m_g   (required) : Path of Gaze Estimation model's xml file.
  5. -m_l   (required): Path of xml file of facial landmarks detection model
  6. -i     (required) : Path of input video file or enter cam for taking input video from webcam.
  7. -o_p   (optional): Path of directory where each output model stats text folder is located in your computer
  8. -vf    (optional): specify flags from m_f, m_l, m_h, m_g e.g. -vf m_f m_l m_h m_g (seperate each flag by space) for visualization of the output of intermediate models 
  
```
python main.py -m_f <Path of xml file of face detection model>-m_l <Path of xml file of facial landmarks detection model>-m_h <Path of xml file of head pose estimation model>-m_g <Path of xml file of gaze estimation model>-i <Path of input video file or enter cam for taking input video from webcam>`
```

## Results

Model loading time(sec), Frame Per Second and Inference Time results can be found [here]()

If we go from FP32 to FP16, accuracy decreases due to lower precision and GPU takes a lot more Model Loading Time than CPU rest all other parameters were considerably same in both the precisions for CPU and GPU.


### Edge Cases
- Multiple People Scenario: If we encounter multiple people in the video frame, it will always use and give results one face even though multiple people detected,
- No Face Detection: it will skip the frame and inform the user
