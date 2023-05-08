# datasetModule

## Installation
Create a Conda environment with python>=3.8, and install pytorch>=1.7 and torchvision>=0.8


Install Segment Anything

'pip install git+https://github.com/facebookresearch/segment-anything.git'

or clone the repository locally and install with

'''
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
'''

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format.
'pip install opencv-python pycocotools matplotlib onnxruntime onnx'


## Getting started

Download the weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)


## Instructions to use the GUI

1. Use the following command to run the GUI
    'python SAM_GUI.py'

2. You'll be first asked to select the weight file. Select the file and click on 'Open'. After this the main GUI window will appear.

2. Click on load image button, select the directory containing the images to be annotated and click on 'Ok'

3. Once an image appears, add prompt points for a particular object using left click. Once you have added enough points (usually 3 to 5 over the image is sufficient), right click to view the segmented object.

4. Repeat step 3 for all the objects that you want to annotate.
Note 1: You can delete the most recent annotation by just pressing 'backspace'
Note 2: You can edit the object category ID at any time by entering the ID in the text box in the bottom-left corner and pessing 'enter'

5. Once you have annotated all the images click on 'Next Image' button

6. Repeat step 3 and 4 for this image.

7. Once all the images have been annotated, click on 'Save Annotations' button
This will generate an 'annotations.json' file in the same directory which contains the annotations in COCO RLE format.