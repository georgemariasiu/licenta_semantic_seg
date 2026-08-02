import cv2, os
from config import param_uavid

image_name = sorted(os.listdir(param_uavid.IMAGE_VAL))[0]
og = cv2.imread(os.path.join(param_uavid.IMAGE_VAL, image_name))

img = cv2.resize(og, (518, 518), interpolation=cv2.INTER_LINEAR)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()