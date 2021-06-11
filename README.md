# HomeSecurityCamera
Home Security Camera project, where raspberry pi micro chip and camera supervise certain location, detects and identifies faces and sends those faces and labels to Azure cloud. A custom android app gets notification from the cloud about people trespassing and also the images of the faces. Basically this is just another smart security camera with custom mobile application and cloud database. See the project wall for more info. 

## Face detection and identification
Two NN models for for the complete person identification. One to detect faces and one to identify. Identification is done by matching created face feature vector to a database of known vectors. Also later applying smart object tracking to get the process lighter.

## Raspberry Pi and Picam
Basic Raspberry pi microchip to do the processing and sending data to cloud. The camera should be as high res. as possible to get accurate detection. 

## Azure cloud
Azure database to store the images of faces and the label data, ie. name of person in the image. Azure sends notification to mobile app that person is detected and also image of the detected person and his/her name. Later the owner can send back eg. a custom message for example "I'm whatching you". 

## Android SafeNest App
A very simple Android mobile application to get notification and view the image of the trespasser. The images and other data are stored in the cloud.  

