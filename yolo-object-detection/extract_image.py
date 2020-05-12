# Not used just for reference

import numpy as np
import cv2
import pyyolo

cap = cv2.VideoCapture('videos/ai')
meta_filepath = "/home/rameshpr/Downloads/darknet_google_server/data/obj.data"
cfg_filepath = "/home/rameshpr/Downloads/darknet_google_server/cfg/yolo-lb.cfg"
weights_filepath = "/home/rameshpr/Downloads/darknet_google_server/backup/yolo-v3.weights"


meta = pyyolo.load_meta(meta_filepath)
net = pyyolo.load_net(cfg_filepath, weights_filepath, False)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    yolo_img = pyyolo.array_to_image(frame)
    res = pyyolo.detect(net, meta, yolo_img)

    for r in res:
        cv2.rectangle(frame, r.bbox.get_point(pyyolo.BBox.Location.TOP_LEFT, is_int=True),
                      r.bbox.get_point(pyyolo.BBox.Location.BOTTOM_RIGHT, is_int=True), (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()