
# High-Throughput Evaluation of Mechanical Exfoliation Using Optical Classification of 2D Materials
A python based image processing software using CuPy for hardware acceleration to classify 2D material thicknesses designed for classification of large datasets very quickly.

The associated publication for this software is located at [INSERT LINK HERE]

This software trains on example 2D material flake images using unsupervised clustering to create a catalog to be used classification of test data images.

Test data images are classified and pixel areas and percentages are exported into a csv file for post-processing and statistical analysis.

This software relies on the CuPy library (requiring use of a dedicated Nvidia GPU) for image processing.



# Usage Instructions:

## Input image data:
- Place full unedited optical microscope images that contain example flakes with known thicknesses to be used for training in the `image_data/training` directory
- Place raw test data optical microscope images into the `image_data/testing` directory

## Specify image areas for training flakes:
- Using the `image-crop.ipynb` notebook:
  - Follow the prompts to select the sub-area of the image with the specific flake to be trained on
    - Output of the notebook will be a JSON object for that flakes location on the test image
    - This JSON object will be copy and pasted as an input in the training step 
  
## Training:
  - Using the `training.ipynb` notebook:
    - If training on a single flake:
      - Paste the JSON object output into the `img_config` input in `main()` function
    - If batch training on multiple flakes:
      - Copy each JSON object into the `batch_data.json` file
  - Follow the prompts to train data and label cluster areas with thicknesses
    - Catalog data for each training flake will be generated in the `outputs/catalogs` directory
  - Run the generate master catalog function to compile all flakes in the `outputs/catalogs` directory to a master catalog located in the `outputs/master_catalog` directory
    - This catalog will be the once used for image classification


## Classification/Testing:
- Using the `testing.ipynb` notebook:
  - Run the cell and all images in the `image_data/testing` will be automatically sequentially classified using the master catalog generated in the previous step
  - Outputs will be generated in the `outputs/test_results` directory and will include:
    - `data` directory with `.json` and `.txt` files that have breakdowns for each image classified
    - `.csv` file containing overall pixel area and layer thickness breakdowns for all images 
    - `images` directory with visualizations of each test image and its classified layer results as a `.png` file
      - `images` enabled by default but can be optionally disabled if unneeded to speed up processing of large image datasets

Final data outputs are `.csv` and `.json` for easy post-processing and statistical analysis using data analytics tools such as Pandas, R, Excel, or MATLAB.




    