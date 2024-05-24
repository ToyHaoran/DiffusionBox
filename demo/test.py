import cv2

# 测试cv2用法

image_path = 'input/000063.JPEG'
output_path = "output/1.jpg"
save = True

bounding_boxes = [(156, 90, 293, 193), (1, 134, 96, 236)]  # 示例边界框
categories = ['domestic_cat: 0.40', 'fox: 0.64']  # 示例类别
RGB_color = [(54,250,6), (26,124,11)]


def draw_bounding_box():
    """
    在图像上绘制边界框和类别标签

    :param image_path: 图像文件的路径
    :param bounding_boxes: 边界框列表，每个边界框是一个包含四个整数的元组 (x_min, y_min, x_max, y_max)
    :param categories: 类别标签列表，每个类别标签对应一个边界框
    """
    # 读取图像
    image = cv2.imread(image_path)

    # image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, tuple(box_color)
    # 设置字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (255, 255, 255)  # 文本颜色为白色
    colors_bg = [(r, g, b)[::-1] for (r, g, b) in RGB_color]  # 边界框和文本背景颜色,将RGB转为BGR
    font_thinkness = 2

    # 绘制每个边界框和类别标签
    for bbox, text, color_bg in zip(bounding_boxes, categories, colors_bg):
        x_min, y_min, x_max, y_max = bbox
        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_bg, font_thinkness)

        # 获取文本框尺寸
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thinkness)

        # 绘制文本框背景
        cv2.rectangle(image, (x_min, y_min), (x_min + text_width, y_min + text_height),
                      color_bg, thickness=-1)  # cv2.FILLED

        # 绘制类别标签
        cv2.putText(image, text, (x_min, int(y_min + text_height + int(font_scale) - 1)), font, font_scale, text_color,
                    font_thinkness)
        break

    if not save:
        # 显示图像
        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 保存结果图像
        cv2.imwrite(output_path, image)


draw_bounding_box()
