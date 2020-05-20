import numpy as np
import cv2
img = cv2.imread('./images/typewriter.jpg')
#print(img.shape)

all_rows = open('./model/synset_words.txt').read().strip().split("\n")

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('./model/bvlc_googlenet.prototxt', './model/bvlc_googlenet.caffemodel')

blob = cv2.dnn.blobFromImage(img, 1, (224,224))

net.setInput(blob)
outp = net.forward()
#print(outp)
idx = np.argsort(outp[0])[::-1][:5]

for(i,id) in enumerate(idx):
    print('{}. {} ({}): Probability {:.3}%'.format(i+1, classes[id], id, outp[0][i]*100))


#Display classes
#for (i,c) in enumerate(classes):
#    if i == 4:
#        break
#    print(i,c)

#display image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
