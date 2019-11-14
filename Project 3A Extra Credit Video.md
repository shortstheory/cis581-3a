# CIS 581 Project 3A Extra Credit: Video Stitching

*Arnav Dhamija and Saumya Shah*

In this portion of the extra credit project, we attempted to stitch videos together to form a panoramic video. 

## Running the Code

Run the following commands in the terminal. Make sure ``ffmpeg` and the appropriate encoders are installed!

To record a video use:

```bash
./record.sh # requres 3 V4L2 webcams connected
```

and then:

```bash
./getframes.sh && ./make_video && ./create_video
```



## Explanation

We started off by trying to use three webcams to record at the same time to get syncrhonized video from multiple angles. In the setup I experimented with, I used a 720p Logitech webcam, my ThinkPad 13's laptop webcam, and a Logitech C920 HD to record video at the same time using the following script called `record.sh`:

```bash
# record.sh
mkdir vids
gst-launch-1.0 v4l2src name=src device=/dev/video0 ! 'image/jpeg,width=(int)640,height=(int)480,framerate=(fraction)30/1' ! jpegdec ! videoconvert ! matroskamux ! filesink location=vids/left.mkv &
gst-launch-1.0 v4l2src name=src device=/dev/video1 ! 'image/jpeg,width=(int)640,height=(int)480,framerate=(fraction)30/1' ! jpegdec ! videoconvert ! matroskamux ! filesink location=vids/middle.mkv &
gst-launch-1.0 v4l2src name=src device=/dev/video2 ! 'image/jpeg,width=(int)640,height=(int)480,framerate=(fraction)30/1' ! jpegdec ! videoconvert ! matroskamux ! filesink location=vids/right.mkv &
sleep 10
killall -9 gst-launch-1.0
```

GStreamer provides a very convenient way to record video to a file and gave me a fine tuned control on the input caps so I could specify the same format and resolution for all webcams. This would give us 300 frames of 480p video.

Unfortunately, I realized after some testing that since the three webcams had vastly different focal lengths and exposure compensation settings, our feature descriptors would be unable to find correspondences between the images. Instead, we were able to obtain left, middle, and right videos from a 360 degree Youtube video which we used for testing our pipelines. These videos can be found in `vids/`.

The next step involved taking the three video and splitting into frames. This was done using `getframes.sh` which would take the three videos and store their JPEGs in `left`, `middle` and `right` directories:

```bash
#getframes.sh
rm *.jpg
rm -rf left
rm -rf middle
rm -rf right
mkdir left
mkdir middle
mkdir right
ffmpeg -i vids/left.mp4 -vf fps=30 %04d.jpg -hide_banner
mogrify -crop 640x330+0+0 *.jpg
mv *.jpg left
ffmpeg -i vids/middle.mp4 -vf fps=30 %04d.jpg -hide_banner
mogrify -crop 640x330+0+0 *.jpg
mv *.jpg middle
ffmpeg -i vids/right.mp4 -vf fps=30 %04d.jpg -hide_banner
mogrify -crop 640x330+0+0 *.jpg
mv *.jpg right
mkdir panovid
```

Now that we have all the frames neatly arranged in folders, we can use `make_video_stitch.py` which takes a frame from each of these three folders, generates the panorama using `mymosaic` and saves the result in the `panovid` directory. We are using 300 frames for this demonstration. Once this is done, I used another script, `create_video.sh` which would take all these frames and stitch them into an MP4 video using ``ffmpeg``:

```bash
#create_video.sh
cd panovid
ffmpeg -i 'output%d.png' -r 10 output.mp4
cp output.mp4 ../
```

...and `output.mp4` is the final output of this pipeline. The video may cause headaches due to all the jumping of the frames. This can be fixed using optical flow for stabilization.




