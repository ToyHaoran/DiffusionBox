import cv2
import numpy as np

# 读取六张图片
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')
img4 = cv2.imread('4.jpg')
img5 = cv2.imread('5.jpg')
img6 = cv2.imread('6.jpg')

save = True

# 检查每张图片的大小是否相同
if not img1.shape == img2.shape == img3.shape == img4.shape == img5.shape == img6.shape:
    # 获取第一张图片的大小
    height, width = img1.shape[:2]
    # 将其他图片缩放为与第一张图片相同的大小
    img2 = cv2.resize(img2, (width, height))
    img3 = cv2.resize(img3, (width, height))
    img4 = cv2.resize(img4, (width, height))
    img5 = cv2.resize(img5, (width, height))
    img6 = cv2.resize(img6, (width, height))

# 拼接第一行的三张图片
row1 = cv2.hconcat([img1, img2, img3])
# 拼接第二行的三张图片
row2 = cv2.hconcat([img4, img5, img6])
# 拼接两行图片
final_image = cv2.vconcat([row1, row2])

if not save:
    # 显示拼接后的图片
    cv2.imshow('Final Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # 保存拼接后的图片
    cv2.imwrite('output/final_image.jpg', final_image)
