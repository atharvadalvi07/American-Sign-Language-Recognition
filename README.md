# American-Sign-Language-Recognition


**Introduction:**

Sign language represents one of the oldest and most instinctual forms of
communication, leading us to devise a real-time technique employing
neural networks for American Sign Language (ASL) finger spelling. The
intention of our project is to aid the people who need assistance with
visual and oral communication. Our proposal introduces a Convolutional
Neural Network (CNN) approach to discern hand gestures in human
activities from camera-captured images. The objective is to identify
hand gestures corresponding to human tasks depicted in these images. By
incorporating hand position and orientation, we derive the training and
testing datasets for the CNN. The hand image undergoes initial
filtration, followed by classification to predict the specific hand
gesture class. Subsequently, the refined images are employed for CNN
training.

American Sign Language is one of the most popular and the oldest sign
languages in the world. Since the only mode of communication for the
disabled people is through sign language, we wanted to devise a
technology that assists them better, this tech will be specially useful
in video conferences and college lectures. the text to speech converter
is also embedded as a feature in the UI interface to provide more
clarity.

In this project we basically focus on producing a model which can
recognize Fingerspelling based hand gestures in order to form a complete
word by combining each gesture. The gestures we aim to train are as
given in the image below.

**Objective:**

The aim is to develop computer software and train a CNN model capable of
processing American Sign Language (ASL) hand gesture images and
converting them into text and audio representations.

**Scope:**

This system holds promise for both individuals who are deaf or mute and
those who are unfamiliar with sign language. Users can communicate
through ASL gestures, and the system will recognize and provide the
output in both text and speech formats, facilitating effective
communication.

**Modules:**

Data Acquisition:

In vision-oriented approaches, the computer\'s webcam serves as the
input mechanism for capturing hand and/or finger-related data. These
methods rely solely on a camera, fostering a seamless interaction
between humans and computers, eliminating the need for additional
devices and subsequently cutting down on expenses. The primary hurdle in
vision-based hand detection revolves around addressing the vast array of
factors influencing the appearance of the human hand. This encompasses
numerous hand movements, varying skin tones, and the diverse factors
like viewpoints, scales, and camera speed at which the scene is
recorded.

Data Pre-processing and Feature Extraction:

In this hand detection approach, our initial step involves identifying
hands within an image captured by a webcam. To achieve this, we utilize
the MediaPipe library, specifically designed for image processing tasks.
Once we successfully pinpoint the hand within the image, we proceed to
define the Region of Interest (ROI). leveraging the capabilities of the
OpenCV library, we enhanced accuracy of our model. Furthermore, we
enhance the image quality by applying Gaussian blur, a filter readily
accessible through the OpenCV library. Following this, we convert the
grayscale image into a binary format using threshold and adaptive
threshold methods.

Our dataset comprises images depicting various signs from different
angles, encompassing sign letters from A to Z.

**Resources Used:**

Python 3.x

OpenCV (cv2)

tkinter

cvzone library

numpy

math

PIL (Python Imaging Library)

ttkthemes

tensorflow (for the classifier model)

The classifier model was trained on a self-made dataset of hand gesture
images using \"https://teachablemachine.withgoogle.com/\".

**Approach:**

Convolutional Neural Network (CNN)

CNN is a class of neural networks that are highly useful in solving
computer vision problems. They found inspiration from the actual
perception of vision that takes place in the visual cortex of our brain.
They make use of a filter/kernel to scan through the entire pixel values
of the image and make computations by setting appropriate weights to
enable detection of a specific feature. CNN is equipped with layers like
convolution layer, max pooling layer, flatten layer, dense layer,
dropout layer and a fully connected neural network layer. These layers
together make a very powerful tool that can identify features in an
image. The starting layers detect low level features that gradually
begin to detect more complex higher-level features

Unlike regular Neural Networks, in the layers of CNN, the neurons are
arranged in 3 dimensions: width, height, depth.

The neurons in a layer will only be connected to a small region of the
layer (window size) before it, instead of all of the neurons in a
fully-connected manner.

Moreover, the final output layer would have dimensions(number of
classes), because by the end of the CNN architecture we will reduce the
full image into a single vector of class scores.

Text To Speech Translation:

The model translates known gestures into words. we have used pyttsx3
library to convert the recognized words into the appropriate speech. The
text-to-speech output is a simple workaround, but it\'s a useful feature
because it simulates a real-life dialogue.

**Challenges:**

In this approach, several limitations become evident. For effective
results, certain conditions need to be met: your hand must be positioned
against a clean, well-lit background. However, the reality of the world
around us often doesn\'t align with these ideal conditions. Backgrounds
can be cluttered, and lighting conditions can be less than optimal.

To address these challenges, we explored various alternative methods and
eventually arrived at an intriguing solution. Initially, we detect the
hand within a frame using the MediaPipe framework, extracting its
landmarks. Subsequently, we draw and connect these landmarks on a plain
white canvas. This innovative approach allows us to work with diverse
backgrounds and lighting scenarios, enhancing the method's robustness
in real-world applications

