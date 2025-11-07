import cv2
from visualization.path import display_gps_path


resolution = 256, 144
output = cv2.VideoWriter("output_videos/first_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 3, resolution)
# use output.write(frame) to add to the video

for i in range(60, 1500, 60):
    if (i <= 840):
        direction = "left"
    else:
        direction = None
    image_path = f"images/256x144/clips/0313-1/{i}/20.jpg"
    overlayed_img = display_gps_path(image_path, direction)
    output.write(overlayed_img)

print("===================================================")
print("--------------- Video Generated! ------------------")
print("===================================================")
