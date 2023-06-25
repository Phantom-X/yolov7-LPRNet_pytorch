from paddleocr import PaddleOCR, draw_ocr, paddleocr

# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(use_angle_cls=True, lang="ch")
# ocr = PaddleOCR(enable_mkldnn=True, use_tensorrt=True, use_angle_cls=False, lang="ch")
# 输入待识别图片路径
img_path = r"LPRNet/data/test/川JK0707.jpg"
# 输出结果保存路径
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
