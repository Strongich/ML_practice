# Task 2. Computer vision. Sentinel-2 image matching
In this task, you will work on the algorithm (or model) for matching satellite images. For the
dataset creation, you can download Sentinel-2 images from the official source [here](https://dataspace.copernicus.eu/browser/) or use our
dataset from [Kaggle](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine). Your algorithm should work with images from different seasons. For this
purpose you need:
* Prepare a dataset for keypoints detection and image matching (in case of using the ML
approach).
* Build / train the algorithm.
* Prepare demo code / notebook of the inference results.

## Dataset 
For this task I've used a dataset from Kaggle that I mentioned above.
## Model selection
For the matching images task I took approach of finding a keypoints of image and than compare them between images. \
Firstly, I've resized all RGB images to 1024x1024 and than saved them as .jpg format instead of .jp2. \
Secondly, I've created [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) detector and extracted features from every image.
Last part was to create a list of images to compare their features. Assuming, that our relation is symmetric (result of comparing image1 and image2 is the same as comparing image2 and image1), our approach will be next:
* we will compare **image1** with **image2**, **image3**, ..., and **last image**
* **image2** with **image3**, **image4**, ... and **last image** 
* **penultimate image** with only **last image**
## Training
As I mentioned above, we have used a SIFT keypoints detector. After all features and keypoints was extracted and saved in the same directory with .jpg image it corresponds, its time to compare them. For this task, I've used knnMatch with 2 nearest neighbors from [BFMatcher](https://docs.opencv.org/4.x/d3/da1/classcv_1_1BFMatcher.html). \
As a result I've had 2 "closest" features to every feature in the first image and then I'm checking if the distance of the first match is less than a second match distance times 0.75 (hyperparameter). If it does, then I consider the match valid and good. As a result I'm returning a count of good matches between images. \
If the amount is larger than 160 (hyperparameter), then I consider that both images are the same and stack them horizontally next to each other to create 1 image. I'm adding lines between matched keypoints and saving them.

## Setup
To use trained model, follow the instructions below:
1. First clone the repository. To do this, open a terminal, go to the directory where you want to clone the project and then enter the command:
```bash
git clone https://github.com/Strongich/ds_intership.git
```
2. Go to folder with project and this task and install virtualenv, write the following command and press Enter:
```bash
cd task2
pip install virtualenv
```
3. Next create a new environment, write the following command and press Enter:
```bash
virtualenv name_of_the_new_env
```
### Example:
```bash
virtualenv immatch
```
4. Next activate the new environment, write the following command and press Enter:
```bash
name_of_the_new_env\Scripts\activate
```
### Example:
```bash
immatch\Scripts\activate
```
5. Write the following command and press Enter:
 ```bash
pip install -r requirements.txt
```
6. Before using it, make sure that you have downloaded images and stored them in folder **data** in task2 folder. \
Then write it in console and press Enter:
```bash
python inference.py
```
It may take some time. \
You will see notes and status of algorithm in console and 2 new folders:
1. **dev_data** - your downloaded dataset, but resized and in .jpg with features to each image.
2. **matched_results** - the output with matched images and lines between keypoints.
## Results
From 1220 combinations of the possible matching pictures we got 41. You can find them at my [Google Drive](https://drive.google.com/drive/u/1/folders/1obTaYc_t9wuczuqPbm7Tm6_vYq2TN4jm). \
 There is few examples of how they looks: \
![Alt text](matched_results/7.jpg) 
![Alt text](matched_results/1.jpg)
