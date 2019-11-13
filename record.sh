mkdir vids
gst-launch-1.0 v4l2src name=src device=/dev/video0 ! 'image/jpeg,width=(int)640,height=(int)480,framerate=(fraction)30/1' ! jpegdec ! videoconvert ! matroskamux ! filesink location=vids/left.mkv &
gst-launch-1.0 v4l2src name=src device=/dev/video1 ! 'image/jpeg,width=(int)640,height=(int)480,framerate=(fraction)30/1' ! jpegdec ! videoconvert ! matroskamux ! filesink location=vids/middle.mkv &
gst-launch-1.0 v4l2src name=src device=/dev/video2 ! 'image/jpeg,width=(int)640,height=(int)480,framerate=(fraction)30/1' ! jpegdec ! videoconvert ! matroskamux ! filesink location=vids/right.mkv &
sleep 10
killall -9 gst-launch-1.0
