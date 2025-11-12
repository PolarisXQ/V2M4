# run python script to extract frames from the video under /public/home/xiaqi2025/video/V2M4/DAVIS

for video in DAVIS/*; do
  video_name=$(basename $video)
  video_name=${video_name%.mp4}
  python extract_frames_white_to_black.py --input $video --output exampleDAVIS/$video_name --prefix '' --start-index 1 --tolerance 20 --interval 1
done


# Use bash to copy the first 10 frames from each folder in exampleDAVIS/* to exampleDAVIS-debug/***
for dir in exampleDAVIS/*; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  mkdir -p "exampleDAVIS-debug/$name"
  # Copy the first 10 files (sorted) from $dir to exampleDAVIS-debug/$name
  files=($dir/*)
  for i in "${files[@]:0:10}"; do
    cp "$i" "exampleDAVIS-debug/$name/"
  done
done

