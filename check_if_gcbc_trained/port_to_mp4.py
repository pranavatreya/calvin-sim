import os
import cv2
import numpy as np

video_seq = np.load("./synthesized_video.npy")

size = (200, 200)
out = cv2.VideoWriter("./synthesized_video.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(16):
    frame = video_seq[i]
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(rgb_img)
out.release()
