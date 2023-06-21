# -*-coding:utf-8-*-
"""
@project:LPRNet_Pytorch
@Author: Phantom
@Time:2023/6/17 2:29
@开发环境：windows 10 + python3.8
@IDE：PyCharm2021.3.1
@Email: 2909981736@qq.com
"""
import torch
from model.LPRNet import build_lprnet
import argparse

parser = argparse.ArgumentParser(description='parameters to train net')
parser.add_argument('--img_size', default=[94, 24], help='the image size')
parser.add_argument('--test_img_dirs', default="./test", help='the test images path')
parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--show', default=True, type=bool, help='show test image and its predict result or not.')
parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')

args = parser.parse_args()

lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=68,
                      dropout_rate=args.dropout_rate)
device = torch.device("cuda:0" if args.cuda else "cpu")
lprnet.to(device)
print("Successful to build network!")
inputs = (1, 3, 24, 94)
input_data = torch.randn(inputs).to(device)
torch.onnx.export(lprnet, input_data, "lpr.onnx")

