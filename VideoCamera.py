class VideoCamera(object):
    #320*240 original
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.cap.isOpened():
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.ret, self.frame = self.cap.read()
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.cap.release()
                return

    def read(self):
        # return the frame most recently read
        return self.ret, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True