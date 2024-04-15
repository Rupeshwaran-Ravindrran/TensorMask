<h1 align="center">TensorMask</h1>

## Description:
This [paper](https://github.com/Rupeshwaran-Ravindrran/TensorMask/blob/master/Report_Paper.pdf) presents a GPU-accelerated real-time facemask detection system using the MobileNetV2 architecture. The proposed system utilizes the power of parallel processing on the GPU to improve the detection speed of facemasks in 
real-time video streams. The MobileNetV2 architecture is selected for its high accuracy and small model size, making it ideal for deployment on mobile devices and embedded systems. The system is trained on a large dataset of facemask 
images and non-facemask images using transfer learning. The experimental results show that the proposed system achieves a high detection accuracy of 95% and a fast-processing speed of 60 frames per second on a NVIDIA RTX 3060 GPU. 
The system can be easily integrated into existing security systems or used to enforce face mask compliance in public places. The proposed system provides an efficient solution for real-time facemask detection with potential applications in public health and safety.


## Frameworks:

- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

## Implementation Novelty:
Our face mask detector doesn't use any morphed masked images dataset and the model is accurate. Owing to the use of MobileNetV2 architecture, it is computationally efficient, thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

## Dataset
The dataset used can be downloaded [here.](https://github.com/Rupeshwaran-Ravindrran/TensorMask/tree/master/dataset)

This dataset consists of __4095 images__ belonging to two classes:
*	__with_mask: 2165 images__
*	__without_mask: 1930 images__

The images used were real images of faces wearing masks. The images were collected from the following sources:

* __Kaggle datasets__ 
* __RMFD dataset__ ([Link](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset))

## Get Started:
1. Clone the repo
```
$ git clone https://github.com/Rupeshwaran-Ravindrran/TensorMask.git
```

2. Change your directory to the cloned repo 
```
$ cd TensorMask
```

3. Create a Python virtual environment named 'test' and activate it
```
$ virtualenv test
```
```
$ source test/bin/activate
```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## Execution:

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
```
2. To detect face masks in real-time video streams type the following command:
```
$ python3 detect_mask_video.py 
```
## Results

After training the MobileNetV2 on the dataset with 20 epochs, it was tested on the testing dataset to acquire an accuracy of over 98%. which indicates that the model is very effective in distinguishing between masked and unmasked faces.

<p align="center">
  <img src="https://github.com/Rupeshwaran-Ravindrran/TensorMask/assets/79376089/811dd4b0-8563-411b-adb3-6857ab357556" alt="GIF" width="500px">
</p>

Precision is the measure of how many of the predicted positive (with_mask or without_mask) cases were actu-ally positive. In this case, the precision for the "with_mask" class is 0.99, meaning that 99% of the pre-dicted cases of with_mask was actually with_mask. Similarly, the precision for the "without_mask" class is 0.97, meaning that 97% of the predicted cases of with-out_mask was actually without_mask.

Recall is the measure of how many of the actual posi-tive (with_mask or without_mask) cases were correctly predicted by the model. In this case, the recall for the "with_mask" class is 0.96, meaning that 96% of the ac-tual cases of with_mask was correctly predicted by the model. Similarly, the recall for the "without_mask" class is 0.99, meaning that 99% of the actual cases of without_mask was correctly predicted by the model.

 F1-score is a measure of the balance between precision and recall, considering both metrics to provide an over-all performance score for each class. In this case, the F1-score for both classes are 0.98, indicating that the model has a good balance between precision and recall for both classes. The support column indicates the number of samples in each class. In this case, there are 1177 samples of with_mask and 1182 samples of with-out_mask in the testing dataset. The accuracy of the model is 0.98, meaning that it correctly predicted the class of 98% of the samples in the testing dataset. 

The macro avg and weighted avg provide an overall per-formance metric for the model, considering the perfor-mance for each class and the number of samples in each class. In this case, both macro avg and weighted avg F1-score and accuracy are 0.98, indicating that the model has performed well overall on the face mask detection task.

Real time performance of the model is evaluated on testing it on a live video source, where we observed the following: 

<p align="center">
  <img src="https://github.com/Rupeshwaran-Ravindrran/TensorMask/assets/79376089/9640f729-d38d-4c8b-89ee-157a2029e5ff" alt="GIF" width="500px">
</p>

In the case of the mobilenetsv2 model for face mask de-tection, the end-to-end latency on the CPU and GPU are 280ms and 28ms, respectively. This difference in per-formance is primarily due to the parallel processing ca-pabilities of the GPU, which can process multiple inputs simultaneously. Our implementation is about 10 times quicker than the reference in this scenario.

---

## Internet of Things Device Setup

### Expected Hardware
* [Raspberry Pi 4 4GB with a case](https://www.canakit.com/raspberry-pi-4-4gb.html)
* [5MP OV5647 PiCamera from Arducam](https://www.arducam.com/docs/cameras-for-raspberry-pi/native-raspberry-pi-cameras/5mp-ov5647-cameras/)

### Getting Started
* Setup the Raspberry Pi case and Operating System by following the Getting Started section on page 3 at `documentation/CanaKit-Raspberry-Pi-Quick-Start-Guide-4.0.pdf` or https://www.canakit.com/Media/CanaKit-Raspberry-Pi-Quick-Start-Guide-4.0.pdf
  * With NOOBS, use the recommended operating system
* Setup the PiCamera
  * Assemble the PiCamera case from Arducam using `documentation/Arducam-Case-Setup.pdf` or https://www.arducam.com/docs/cameras-for-raspberry-pi/native-raspberry-pi-cameras/5mp-ov5647-cameras/
  * [Attach your PiCamera module to the Raspberry Pi and enable the camera](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/2)

### Raspberry Pi App Installation & Execution

> Run these commands after cloning the project

| Commands                                                                                                                     | Time to completion |
|------------------------------------------------------------------------------------------------------------------------------|--------------------|
| sudo apt install -y libatlas-base-dev liblapacke-dev gfortran                                                                | 1min               |
| sudo apt install -y libhdf5-dev libhdf5-103                                                                                  | 1min               |
| pip3 install -r requirements.txt                                                                                             | 1-3 mins           |
| wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.4.0-cp37-none-linux_armv7l_download.sh" | less than 10 secs  |
| ./tensorflow-2.4.0-cp37-none-linux_armv7l_download.sh                                                                        | less than 10 secs  |
| pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl                                                                     | 1-3 mins           |

---

## License

This project is licensed under the [MIT License](LICENSE), allowing for open collaboration and usage.



