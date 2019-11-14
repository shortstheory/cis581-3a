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
