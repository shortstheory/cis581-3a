#create_video.sh
cd panovid
ffmpeg -i 'output%d.png' -r 5 output.mp4
cp output.mp4 ../

