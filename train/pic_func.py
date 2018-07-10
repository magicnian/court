import cv2
import os

src_dir = 'H:\\document\\ocr\\src\\shixin2'
dst_dir = 'H:\\document\\ocr\\dst-test'

files = os.listdir(src_dir)

for name in files:
    im = cv2.imread(os.path.join(src_dir, name))
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(dst_dir, name), thresh)
print('done')