import cv2
def process_this_video(inputvideopath, outputvideopath):
    video = cv2.VideoCapture(inputvideopath)
    if not video.isOpened():
        print("Can't open the video")
        return
    

    """
    Here are the video properties, which will then be used to
    create the dimensions for the output video
    """
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = cv2.VideoWriter(outputvideopath, codec, fps, (width, height))

    """
    I've created the output video's dimensions, now I will fill it in
    one frame at a time
    """

    for frameIndex in range (total_frames):
        successfullyRead, frame = video.read()
        if not successfullyRead:
            print("error reading the frame")
            break

        processed_frame = detect_curved_lanes(frame)

        output_video.write(processed_frame)

        if frameIndex & 100 == 0:
            print(f"Processed {frameIndex} frames")

    # save the file
    video.release()
    output_video.release()
    print(f"Output saved to {outputvideopath}")