import cv2
import numpy as np
import argparse
from collections import deque


frame_size =(720,1638)
class AverageBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.shape = None
        self.non_zero_buffer = deque(maxlen = 5)
        

    def apply(self, frame):
        self.shape = frame.shape
        
        data = np.asarray(frame,dtype='int32')
        sum = data.sum()
        if(sum > 1000000 ):
            self.non_zero_buffer.append(frame)
        if(len(self.non_zero_buffer) != 0 and sum < 1000000 ):
            self.buffer.append(self.non_zero_buffer[-1])
        else:
            self.buffer.append(frame)

    def get_frame(self):
        mean_frame = np.zeros(self.shape, dtype='float32')
        for item in self.buffer:
            mean_frame += item
        mean_frame /= len(self.buffer)
        return mean_frame.astype('uint8')


# class WeightedAverageBuffer(AverageBuffer):
#     # def get_frame(self):
#     #     mean_frame = np.zeros(self.shape, dtype='float32')
#     #     i = 0
#     #     for item in self.buffer:
#     #         i += 4
#     #         mean_frame += item*i
#     #     mean_frame /= (i*(i + 1))/8.0
#     #     return mean_frame.astype('uint8')



def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()
    old_frame = cv2.resize(old_frame, (0, 0), fx = 0.4, fy = 0.4)
    val = old_frame.shape
    print(val)

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    average_buffer = AverageBuffer(5)
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    # output = cv2.VideoWriter('output_video_from_file.mp4', fourcc, 24.0, frame_size)
    count =0
    # weighted_buffer = WeightedAverageBuffer(5)
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        

        new_frame = cv2.resize(new_frame, (0, 0), fx = 0.4, fy = 0.4)

        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        min = np.array([30,0,55],np.uint8)
        max = np.array([100,255,255],np.uint8)
        bgr = cv2.inRange(hsv,min,max)
        # print("shape: ", hsv1.shape)
        # bgr = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
        
        

        # frame_copy = cv2.resize(frame_copy, (960, 540))
    
        # cv2.imshow("frame", frame_copy)


        # bgr = cv2.add(bgr,frame_copy)
        # print('value of the frame',np.sum(bgr,dtype = np.uint8))
        
        # data = np.asarray(bgr,dtype='int32')
        # sum = data.sum()
        # print('sum', sum)
        cv2.imshow("optical flow", bgr)
        # No_image_threshold = 
        kernel = np.ones((12,12), np.uint8)
        bgr = cv2.morphologyEx(bgr, cv2.MORPH_OPEN, kernel) 

        bgr = cv2.dilate(bgr, kernel, iterations=2)      
        bgr = bgr.astype('float32')
        average_buffer.apply(bgr)
        # weighted_buffer.apply(bgr)
        # cv2.imshow("Average", average_buffer.get_frame())
        # cv2.imshow("Weighted average", weighted_buffer.get_frame())

        t, thresholded = cv2.threshold(average_buffer.get_frame(), 90, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Thresholded", thresholded)
        contours, hierarchy = cv2.findContours(image=thresholded, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=frame_copy, contours=contours, contourIdx=-1, color=(0, 255,0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Final", frame_copy)
        count +=1
        cv2.imwrite('frame'+str(count)+'.jpg',frame_copy)
        # output.write(frame_copy)

        #24
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame

        
