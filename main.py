import os
import time
import torch
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from LPRNet.data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from LPRNet.model.LPRNet import build_lprnet
from torchvision import transforms
from yolo_carid import YOLO

transformsTotensor = transforms.Compose([
    transforms.ToTensor()
])


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=True, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./LPRNet/weights/Final_LPRNet_model.pth',
                        help='pretrained base model')
    parser.add_argument('--mode', default='dir_predict', help='识别模式，predict,video,dir_predict等')
    parser.add_argument('--count', default=False, help='是否计数')
    parser.add_argument('--video_path', default="img/test2.mp4", help='视频路径，0是摄像头')
    parser.add_argument('--video_save_path', default="./output/2023-6-26.mp4", help='检测视频保存路径')
    parser.add_argument('--video_fps', default=10.0, help='用于保存的视频的fps')
    parser.add_argument('--test_interval', default=25.0,
                        help='用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。')
    parser.add_argument('--fps_image_path', default="img/street.jpg", help='用于指定测试的fps图片')
    parser.add_argument('--dir_origin_path', default="img/dir/", help='指定了用于检测的图片的文件夹路径')
    parser.add_argument('--dir_save_path', default="output/dir/", help='指定了检测完图片的保存路径')
    parser.add_argument('--heatmap_save_path', default="model_data/heatmap_vision.png",
                        help='heatmap_save_path   热力图的保存路径，默认保存在model_data下')
    args = parser.parse_args()

    return args


args = get_parser()


def pretreatment(img):
    img = cv.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img


def initlprnet():
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=False, class_num=len(CHARS),
                          dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        return lprnet
    else:
        return None


lprnet = initlprnet()
if lprnet is None:
    print("[Error] Can't found pretrained mode, please check!")
else:
    print("load pretrained model successful!")


def recognize(image, result_set):
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    img_set = []
    for r in result_set:
        img_set.append(image[r[1]:r[3], r[2]:r[4]])
    img_set = list(map(pretreatment, img_set))
    img_set = np.array(img_set)
    img_set = torch.from_numpy(img_set)
    if args.cuda:
        img_set = Variable(img_set.cuda())
    else:
        img_set = Variable(img_set)
    print(img_set.shape)

    prebs = lprnet(img_set)
    # greedy decode
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    carid_set = []
    for label in preb_labels:
        carid = ""
        for l in label:
            carid += CHARS[l]
        carid_set.append(carid)
    # print(carid_set)
    tuple_set = zip(result_set, carid_set)
    return tuple_set


def show(image, tuple_set):
    image = cvImgAddText(image, tuple_set)
    cv.imshow("test", image)
    cv.imwrite("./output/1.jpg", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cvImgAddText(img, tuple_set, textColor=(255, 0, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("LPRNet/data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    for pos, carid in tuple_set:
        draw.text((pos[2], pos[1] - 25), carid, textColor, font=fontText)
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def detect():
    mode = args.mode
    count = args.count
    video_path = args.video_path
    video_save_path = args.video_save_path
    video_fps = args.video_fps
    test_interval = args.test_interval
    fps_image_path = args.fps_image_path
    dir_origin_path = args.dir_origin_path
    dir_save_path = args.dir_save_path
    heatmap_save_path = args.heatmap_save_path

    yolo = YOLO()

    if mode == "predict":
        while True:
            imgpath = input('Input image filename:')
            # imgpath = "img/1.jpg"
            try:
                image = Image.open(imgpath)
            except:
                print('Open Error! Try again!')
                continue
            else:
                result_set = yolo.detect_image(image, count=count)
                if result_set is None:
                    image.show()
                else:
                    tuple_set = recognize(image, result_set)
                    show(image, tuple_set)

    elif mode == "video":
        capture = cv.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
            out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测得到box结果集
            result_set = yolo.detect_image(frame)
            # RGBtoBGR满足opencv显示格式
            if result_set is None:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            else:
                tuple_set = recognize(frame, result_set)
                frame = cvImgAddText(frame, tuple_set)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv.putText(frame, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow("video", frame)
            c = cv.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv.destroyAllWindows()
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    elif mode == "dir_predict":

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                result_set = yolo.detect_image(image)
                if result_set is None:
                    r_image = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)
                else:
                    tuple_set = recognize(image, result_set)
                    r_image = cvImgAddText(image, tuple_set)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv.imwrite(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), r_image)
                # r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'dir_predict'.")


if __name__ == '__main__':
    detect()
