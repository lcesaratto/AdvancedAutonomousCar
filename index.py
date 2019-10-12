import cv2
import numpy as np


def line_keeping():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('Videos/WhatsApp Video 2019-10-12 at 6.19.29 PM(2).mp4')

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, flipCode=-1)
        
        if ret:
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([255, 255, 90]) #255, 255, 90
            mask = cv2.inRange(hsv, lower_black, upper_black)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            dif_gray = cv2.GaussianBlur(mask, (3, 3), 0) #(mask, (5, 5), 0)
            canny = cv2.Canny(dif_gray, 1, 500,) # 25, 175)
    
            lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 20, minLineLength=5, maxLineGap=25) #(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=150)
            # Draw lines on the image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # Show result
            #cv2.imshow("Result Image", frame)

            # Display the resulting frame
            #cv2.imshow('Frame', canny)
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):  # 25fps
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    line_keeping()
