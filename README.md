# Mask-recognition
GitHub README for Face Mask Detection Project

Face Mask Detection Using YOLO and CNN

Overview

This project implements a Face Mask Detection system using a combination of Convolutional Neural Networks (CNN) and YOLO (You Only Look Once), designed to detect whether people are wearing face masks in images. The primary goal is to improve face mask detection accuracy in various scenarios, particularly when dealing with multiple faces and complex environments.

The dataset used in this project is provided by Kaggle: Face Mask Detection. This dataset contains thousands of images categorized as “with mask,” “without mask,” and “improperly worn mask.”

The project also references an example code from Kaggle: YOLOv7 Face Mask Detection.

Features

	•	Mask Detection with CNN: A baseline model using CNN for detecting masks on faces in simpler images.
	•	YOLO Integration: YOLO is used for multi-face detection in more complex scenarios, improving the detection of masks in crowded or multi-object images.
	•	Loss and Accuracy Optimizations: The implementation introduces dynamic learning rate adjustments and stability techniques to address training loss bumps and oscillations.

Dataset

The dataset contains the following classes:

	•	With Mask: Images of people wearing masks properly.
	•	Without Mask: Images of people not wearing masks.
	•	Improperly Worn Mask: Images where people are wearing masks incorrectly.

You can download the dataset from Kaggle: Face Mask Detection Dataset.

Setup and Installation

	1.	Clone the Repository:

git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection


	2.	Install Dependencies:
Ensure you have the necessary packages installed by running:

pip install -r requirements.txt


	3.	Prepare Dataset:
Download the Face Mask Detection dataset from Kaggle and copy the images into the dataset folder:

!mkdir -p Dataset/FaceMask/images
!mkdir -p Dataset/FaceMask/labels
!cp -rf /kaggle/input/face-mask-detection/images/* Dataset/FaceMask/images


	4.	Run YOLOv7 Setup:
Clone the YOLOv7 repository and install necessary dependencies:

git clone https://github.com/rkuo2000/yolov7.git
cd yolov7
pip install -r requirements.txt
pip uninstall wandb -y
pip install split-folders



Usage

	1.	Training the CNN Model:
	•	Load the dataset and preprocess the images (resize, normalize, etc.).
	•	Train the CNN model on the dataset using the provided cnn_recognition.py script.
	2.	Using YOLO for Detection:
	•	YOLOv7 is used to detect masks in more complex images (multiple faces).
	•	You can switch between the CNN and YOLO models based on the complexity of the images.
	3.	Evaluation:
	•	Evaluate the performance of both models using metrics such as accuracy, precision, recall, and F1 score.
	•	YOLO is particularly suited for real-time detection in complex environments.

Results

The CNN model achieved an accuracy of 93% on simpler datasets, while YOLO improved detection in more complex scenarios, with a significant reduction in loss fluctuation and better multi-face detection.

Key insights:

	•	CNN: Works well for simpler cases with fewer faces and clear visibility.
	•	YOLO: Performs significantly better in detecting multiple faces and handling occlusions.

Future Work

	•	Improve YOLO Detection: Enhance the detection of improperly worn masks, which still presents challenges.
	•	Real-time Application: Implement the model for real-time mask detection in public places.
	•	Model Fine-tuning: Further tune the hyperparameters and increase the robustness of the model for different lighting and environmental conditions.

References

	•	Face Mask Detection Dataset: Kaggle - Face Mask Detection
	•	YOLOv7 Example Code: YOLOv7 Face Mask Detection on Kaggle

License

This project is licensed under the MIT License. See the LICENSE file for more details.

This README file provides a clear structure of your project, including key components such as setup, usage, results, and future work. Let me know if you want to add any further details or adjustments!
