# CIS 581 Project 1B

## About

Gradient blending is an image processing technique to superimpose one image in the other making the superimposed image look part of the target image. In this project we have used the Poisson Image Editing method for creating a seamless cloning tool.

## Running the code

Run

```
python3 demo.py
```

This will superimpose a picture of the Pokemon, Eevee, with a table. The result of this operation will be saved in an image `eevee.png` in the same folder.

If you wish to create the mask from scratch with these two images, run the following instead. The offsets and images can be modified in the script:

```
python3 full_demo.py
```


## Results

The results of gradient blending of the minion and Benjamin Franklin are available in the `results/` directory.

## Techniques Used

In this approach, we first sequentially number the indices of the source image with respect to its provided mask in `getIndexes`. `getSolutionVect` produces the RHS for the equation Ax = b for each color channel by applying the laplacian filter to the source image and adding the background components of the target image. Once these steps are completed we can generate the doctored image by solving the above linear equation and combining the channels will give us our final output.

The solution vectors for the red, blue, and green channels contain value outside of the domain of colors, that is it returns pixel both negative intensities and intensities greater than 255. For fixing this, the pixel intensities are normalized between 0 and 255 for each channel with respect to the overall minimum and maximum pixel intensities. This solves the problem of wrongly colored images in the output.
