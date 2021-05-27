# importing libraries
import cv2
import numpy as np
import datetime


cap = cv2.VideoCapture(0)  # VideoCapture is used to capture video from webcam
# here '0' indicates that the video will be captured from the first webcam on your device


# taking input point
# this is used to capture very first frame of the video and we will use it later on 
# in calcOpticalFlowPyrLK() function
# for this we have created gray input image with blur of (4,4) kernel


_, inp_img = cap.read()
inp_img = cv2.flip(inp_img, 1)
inp_img = cv2.blur(inp_img, (4, 4))
gray_inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

# tracking starts here

# the old_pts is the initialized window which we see when the code is run
old_pts = np.array([[350, 180], [350, 350]], dtype=np.float32).reshape(-1, 1, 2)

# creating backup of old points and storing it in backup_img
backup = old_pts.copy()
backup_img = gray_inp_img.copy()

# Output windows which is blank and black in color which will show text
outp = np.zeros((480, 640, 3))

# variable
ytest_pos = 40  # -----> to print the text in various positions in output window
###############


# creating an infinite loop so that user can exit whenever he/she wants
while True:
    _, new_inp_img = cap.read()  # reading the frame
    new_inp_img = cv2.flip(new_inp_img, 1)  # flipping the frame
    new_inp_img = cv2.blur(new_inp_img, (4, 4))
    new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)

    # the function returns values of the variables on the LHS
    # parameters are the old gray img, new gray img, collection of old points
    # None is used as we don't have any info about the new points at the time
    # Here max_level which has value 0 indicates that the resolution is as the original one
    # criteria has epsilon and iterations count, more the value, better is the result
    # it means that once the accuracy has reached the value of epsilon or the max no of iterations
    # algorithm stops working
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_inp_img,
                                                    new_gray,
                                                    old_pts,
                                                    None, maxLevel=0,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                              15, 0.8))

    # creating boundaries for the points which we are tracking
    # this prevents any point from getting outside the window    

    if new_pts.ravel()[0] >= 600:
        new_pts.ravel()[0] = 600
    if new_pts.ravel()[1] >= 350:
        new_pts.ravel()[1] = 350
    if new_pts.ravel()[0] <= 20:
        new_pts.ravel()[0] = 20
    if new_pts.ravel()[1] <= 150:
        new_pts.ravel()[1] = 150
    if new_pts.ravel()[2] >= 600:
        new_pts.ravel()[2] = 600
    if new_pts.ravel()[3] >= 350:
        new_pts.ravel()[3] = 350
    if new_pts.ravel()[2] <= 20:
        new_pts.ravel()[2] = 20
    if new_pts.ravel()[3] <= 150:
        new_pts.ravel()[3] = 1507

    # drawing line from (x,y) to (a,b) of thickness 15 of red color
    x, y = new_pts[0, :, :].ravel()
    a, b = new_pts[1, :, :].ravel()
    cv2.line(new_inp_img, (x, y), (a, b), (0, 0, 255), 15)

    cv2.imshow("OUTPUT", new_inp_img)

    # if x> 400 and x>550, the text is shown in the output window
    # i.e. if the user goes out of the room, new pts will have backup points
    # new img will be assigned with backup img

    if new_pts.ravel()[0] > 400 or new_pts.ravel()[2] > 400:
        if new_pts.ravel()[0] > 550 or new_pts.ravel()[2] > 550:
            new_pts = backup.copy()
            new_inp_img = backup_img.copy()
            ytest_pos += 40
            cv2.putText(outp, "Gone at {}".format(datetime.datetime.now().strftime("%H:%M:%S")), (10, ytest_pos),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))

    elif new_pts.ravel()[0] < 200 or new_pts.ravel()[2] < 200:
        if new_pts.ravel()[0] < 50 or new_pts.ravel()[2] < 50:
            new_pts = backup.copy()
            new_inp_img = backup_img.copy()
            ytest_pos += 40
            cv2.putText(outp, "Came at {}".format(datetime.datetime.now().strftime("%H:%M:%S")), (10, ytest_pos),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

    cv2.imshow('final', outp)
    gray_inp_img = new_gray.copy()
    old_pts = new_pts.reshape(-1, 1, 2)

    # if user presses esc key and have waited 1 ms, infinite loop is broken
    if cv2.waitKey(1) & 0xff == 27:
        break

# when pressed  exit key, it will destroy all the windows and will release the camera 
cap.release()
cv2.destroyAllWindows()
