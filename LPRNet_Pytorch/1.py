# -*-coding:utf-8-*-
"""
@project:LPRNet_Pytorch
@Author: Phantom
@Time:2023/6/19 15:34
@开发环境：windows 10 + python3.8
@IDE：PyCharm2021.3.1
@Email: 2909981736@qq.com
"""
import time

a = [i for i in range(10000000)]
start = time.time()
b = list(map(lambda x: x * 2, a))
print(time.time()-start)

start = time.time()
for i in range(10000000):
    b = a[i]*2
print(time.time()-start)
