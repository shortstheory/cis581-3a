# CIS 581 Project 3A

**Team Members:** Arnav Dhamija and Saumya Shah

This project has been documented in the included PDF.

## Structure

The code can be divided into the following files:

* `corner_detector.py`: uses the Harris corner detector for keypoints
* `anms.py`: does adaptive non-maximum suppression to obtain a balanced distribution of N keypoints
* `feat_desc.py`: create a 64xN matrix of feature vectors for the keypoints and image
* `feat_match.py`: matches keypoints between two images using FLANN
* `ransac_est_homography.py`: uses RANSAC to find good inliers and to construct a homography matrix
* `get_homography.py`: runs all of the above steps to find a homography from the left image to the right image
* `mymosaic.py`: alpha blends and feathers the images according to the provided homographies

Images we used can be found in `images/`, and different approaches we experimented with can be found in `variations/`. The `results/` folder has a subfolder for each set of images with intermediate outputs with each step.

## Running the Code

The code can be run in two ways:

```
python3 demo.py
```

will create a panorama of Shoemaker's Green from the images in the `images/` folder.

The other option is to open `demo.ipynb` in VS Code or Jupyter Notebooks in a browser for an interactive demo of the project.