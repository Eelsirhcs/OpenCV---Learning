import numpy as np
import cv2

cap = cv2.VideoCapture('./images/shore.mov')

all_rows = open('./model/synset_words.txt').read().strip().split("\n")

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('./model/bvlc_googlenet.prototxt', './model/bvlc_googlenet.caffemodel')

if cap.isOpened() == False:
    print('Cannot open file or video stream')

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1, (224,224))

    net.setInput(blob)

    outp = net.forward()

    r = 1
    for i in np.argsort(outp[0])[::-1][:5]:
        txt = ' "%s" probability "%.3f" ' % (classes[i], outp[0][i] * 100)
        cv2.putText(frame, txt, (0,25 + 40*r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        r+=1

    if ret == True:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
