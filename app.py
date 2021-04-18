from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

# initiating Tinker
from pafy import pafy

root = Tk()

# setting the size of the gui
root.geometry("720x480")

# setting a minimum size
root.minsize(width=640, height=480)

# Gui Title
root.title("Intelligent Video Surveillance using YOLO Algorithm")


# ----------------------------------------------IMAGE DETECTION--------------------------------------------------------

def object_detection_via_image():
    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    url = str(image_get.get())
    if url == '0' or url=='':
        img = cv2.imread("test2.jpg")


    else:
        img = cv2.VideoCapture(url)
        if (img.isOpened()):
            ret, img = img.read()

    img = cv2.resize(img, (852, 480), interpolation=cv2.INTER_AREA)

    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + ' ' +str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

    # resized = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Gui Heading
f1 = Frame(root, bg='grey', borderwidth=2).pack(anchor="nw")

image_get = Entry(root)
image_get.pack(pady=20)
desc = Label(root, text="Enter the image url above!").pack()

# Button for activating Image Object Detection
button_1 = Button(f1, fg="teal", text="Object Detection in a Image", command=object_detection_via_image).pack()


# ------------------------------------------------MOVEMENT DETECTION----------------------------------------------------
def movement():
    url = str(media.get())

    if url[-5:] == 'video':
        cap = cv2.VideoCapture(url)

    elif url == '0' :
        cap = cv2.VideoCapture(0)

    elif url == '':
        cap = cv2.VideoCapture("test.mp4")
    else:
        try:
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")
            cap = cv2.VideoCapture()
            cap.open(best.url)
        except:
            print("An incorrect youtube link or an invalid source might be uploaded!\nInstead watch the prototype.")
            cap = cv2.VideoCapture("test.mp4")

    # cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGRA2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame1, contours, -1, (0, 255, 0) ,2)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 700:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status : {}".format("Movement Detected"), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 3)

        cv2.imshow("Feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


f2 = Frame(root, bg='grey', borderwidth=2).pack(anchor="nw")

media = Entry(root)
media.pack(pady=20)
desc = Label(root, text="Supports Youtube Live video too!").pack()
desc = Label(root, text="if using Ip cam add '/video' at the end of the url!").pack()

# Button for activating Movement Detection
button_2 = Button(f2, fg="teal", text="Surveillance", command=movement).pack()


# -------------------------------------------------LIVE OBJECT DETECTION----------------------------------------------

def live_video():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    classes = []

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # loading image
    url = str(video_get.get())

    if url[-5:] == 'video':
        cap = cv2.VideoCapture(url)

    elif url == '0':
        cap = cv2.VideoCapture(0)

    elif url == '':
        cap = cv2.VideoCapture("test.mp4")

    else:
        try:
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")
            cap = cv2.VideoCapture()
            cap.open(best.url)
        except:
            print("An incorrect youtube link or an invalid source might be uploaded! \nInstead watch the prototype.")
            cap = cv2.VideoCapture("test.mp4")

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()

        height, width, channels = frame.shape

        # reducing 416 to 320
        blob = cv2.dnn.blobFromImage(frame,
                                     0.00392,
                                     (320, 320),
                                     (0, 0, 0),
                                     True,
                                     crop=False)
        net.setInput(blob)
        outs = net.forward(outputlayers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS:" + str(round(fps, 2)),
                    (10, 50),
                    font,
                    2,
                    (0, 0, 0), 1)
        cv2.imshow("Image", frame)

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


f3 = Frame(root, bg='grey', borderwidth=2).pack(anchor="nw")

video_get = Entry(root)
video_get.pack(pady=20)

desc = Label(root, text="Supports Youtube Live video too!").pack()
desc = Label(root, text="if using Ip cam add '/video' at the end of the url!").pack()

# Button for activating Image Object Detection
button_3 = Button(f3, fg="teal", text="Object Detection in a Live Video", command=live_video).pack()

# Image Object Detection end here

# image
f2 = Frame(root, borderwidth=5, bg="grey", relief=SUNKEN).pack()

root.mainloop()
