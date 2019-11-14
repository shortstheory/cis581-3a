CIS 581 Project 3A Extra credit: Cylindrical Panorama
- Arnav Dhamija and Saumya Shah


This project has been documented in the included PDF.

## Structure

The code can be divided into the following files:

* `corner_detector.py`: uses the Harris corner detector for keypoints.
* `anms.py`: does adaptive non-maximum suppression to obtain a balanced distribution of N keypoints
* `feat_desc.py`: create a 64xN matrix of feature vectors for the keypoints and image.
* `feat_match.py`: matches keypoints between two images using FLANN. This function also returns dMatch, which is a list of objects which are used for generating the RANSAC plots for the matching in the later code. This may be removed when performing the individual function testing.
* `get_cylindrical.py`: Generates the cylindrical projections of the passed images and stores them as 'cylin1.jpeg','cylin2.jpeg','cylin3.jpeg' respectively in the working directory.
* `ransac_est_homography.py`: uses RANSAC to find good inliers and to construct a homography matrix
* `utilities.py`: the helper file which contains functions for warping images and a wrapper function `get_homography` which executes all of the above functions.
* `mymosaic.py`: alpha blends and feathers the images according to the provided homographies.
* `record.sh`: records videos from 3 connected webcams at once.
* `get_frames.sh`: splits the videos found in the `vids/` folder into its individual frames
* `stitch_video_frames.py`: creates a video from the frames we found by running the above shell scripts
* `create_video.sh`: creates a video from the frames generated in the previous step

Images we used can be found in `images/`, and different approaches we experimented with can be found in `variations/`. The `results/` folder has a subfolder for each set of images with intermediate outputs with each step. The `results/`, folder contains the resulting images obtained at each step for each of our images. The `results/shoemaker/` folder contains the results from the images of the Franklin Field.

### Extra Credit

Results for the cyldrical projection and the video projection can be found in `extra_credit_cylindrical_results` and `extra_credit_video_results` respectively.

## Running the Code

### Extra Credit

For cylindrical projection, run:
```
python3 cylindrical_demo.py
```
For video projection, run:
```
./getframes.sh 
./stitch_video_frames.py
./create_video.sh
```

### Regular Credit

The code can be run in two ways:

```
python3 demo.py
```

will create a panorama of Shoemaker's Green from the images in the `images/` folder.
(Please change the filename and path appropriately for testing on other images)

The other option is to open `demo.ipynb` in VS Code or Jupyter Notebooks in a browser for an interactive demo of the project, generating plots for each intermediate step.
