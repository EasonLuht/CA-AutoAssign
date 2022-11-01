# -*- coding:utf-8 -*-
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import time


def main():
    parser = ArgumentParser()
    parser.add_argument('imgpath', help='Image path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    basepath = args.imgpath
    imglist = os.listdir(basepath)[0:1200]  # 前10张预热，后10张收尾，只计算推理中间100张的时间
    print('capacity :', len(imglist))
    for index, img in enumerate(imglist):
        if index == 100:
            print('start count')
            start = time.time()
        _ = inference_detector(model, os.path.join(basepath, img))
        if index == 1100:
            print('end count')
            end = time.time()

    print("--------------------------------------------------------")
    total = end - start
    print("total:", total)
    print('fps===>', 1000 / total)
    print('over')


if __name__ == '__main__':
    main()
