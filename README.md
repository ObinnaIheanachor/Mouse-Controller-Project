# Computer Pointer Controller Project

In this project, I have used an Intel® OpenVINO Gaze Detection model to control the mouse pointer of my computer. The final output, which is the x and y coordinates of the eye gaze, from the combined models is then fed to a mouse controller which moves the mouse pointer to the given coordinates.

This project comprises 4 models which are face detection, facial landmark recognition, head pose estimation and gaze estimation which is used to produce 
the x and y values that is used to control the mouse. In the face detection model we capture the face in the frame and the cropped face is sent as input to the Landmark detection model and Headpose estimation models. Landmark detection model sends the cropped left and right eye to the Gaze estimation model while the Head pose estimation model sends the head pose angles. The gaze estimation model processes these inputs and use it to control the mouse pointer.
## Project Set Up and Installation1) Download and install Intel OpenVino Toolkit2) Clone/download this repo.3) Create a virtual environment You can use conda:`conda create --name myenv python=3.6`4) Install the required packages in the requirements file using:`pip install -r requirements.txt`
5) Initialize the openvino toolkit by ```"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"```
6) Download the required models using the following commands:1st Face Detection Model:-
python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "face-detection-adas-binary-0001"
2nd Facial Landmarks Detection Model:-
python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "landmarks-regression-retail-0009"
3rd Head Pose Estimation Model:-
 python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "head-pose-estimation-adas-00     
4th Gaze Estimation Model:-   
    python "C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py" --name "gaze-estimation-adas-0002"
## Demo 1) Change the directory to src directory of project cd <project-path>/src 2) Run the main.py file Example: python main.py -m_f "C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\deployment_tools\open_model_zoo\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -m_l "C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\deployment_tools\open_model_zoo\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -m_h "C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\deployment_tools\open_model_zoo\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -m_g "C:\Program Files (x86)\IntelSWTools\openvino_2020.2.117\deployment_tools\open_model_zoo\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002" -o_p "C:\Users\HELLO\Desktop\DS\Nanodegrees\Intel Edge Nanodegree\starter\starter\src\output_path" -i "C:\Users\HELLO\Desktop\DS\Nanodegrees\Intel Edge Nanodegree\starter\starter\bin\demo.mp4"
## command line options 
`python main.py -m_f <Path of xml file of face detection model>-m_l <Path of xml file of facial landmarks detection model>-m_h <Path of xml file of head pose estimation model>-m_g <Path of xml file of gaze estimation model>-i <Path of input video file or enter cam for taking input video from webcam>` 

The project is made up of the following command line arguments 
- the first is'-m_l' and represents the directory or location where the face model is located -The second is denoted as '-m_l' and represents the directory where the lamdmark model is located 
- Third is'-m_h' and represents the directory where the headpose model is located
- Fourth is '-m_g' and represents the directory where the gaze model is located
-The fifth is denoted as '-d' and represents the directory where the gaze model is located
- Sixth is '-i' and represents the directory where the gaze model is located
- Seventh is '-o_p' and represents the directory where each output model stats text folder is located
- Eighth is denoted as '-vf' and specifies flags from m_f, m_l, m_h, m_g  e.g. -vf m_f m_l m_h m_g (seperate each flag by space) for visualization of the output of intermediate models
-The eight is denoted as '-modelAr' to view the model architecture.

File structure of the project:
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
1. main.py : Used for running the app. 2. face_detection.py: Used for preprocessing video frame, perform inference and detect the face, then postprocess the outputs. 3. facial_landmark_detection.py: Takes detected face as input, preprocesses it, performs inference on it and detects the eye landmarks, then postprocess outputs. 4. head_pose_estimation.py: Takes the detected face as input, preprocesses it, performs inference on it and detects head position. 5. gaze_estimation.py: Takes the left eye, right eye, head pose angles as inputs, preprocesses it, performs inference and predicts the gaze vector, postprocess the outputs. 6. input_feeder.py: Contains InputFeeder class which initialize VideoCapture as per the user argument and returns frames. 7. mousecontroller.py: Contains MouseController class. 8. output_path folder: contains inference results across all the models and different devices
\bin folder contains demo video which is used for testing the app.
## Benchmarks
Model loading time(sec):
	Face model: ~0.35s
	Gaze model: ~0.26s
	headpose: ~0.21s
	landmark: ~0.21s
FPS :
	Face model: ~3
	Gaze model: ~24
	headpose: ~30
	landmark: ~51
Model inference time(sec):0.
	Face model: ~0.26s
	Gaze model: ~0.04s
	headpose: ~0.3s
	landmark: ~0.24s
Generally FP32 precision gives better accuracy when compared to FP16 due to loss in precision.
### Edge Cases
If there are multiple people in the frame, results for only one face is returned.
If there is no face detected, it will skip the frame.