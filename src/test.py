# from share4v.mm_utils import get_model_name_from_path
# from share4v.eval.run_share4v import eval_model

# debug = False


# model_path = "Lin-Chen/ShareGPT4V-7B"
# prompt = "What is the most common catchphrase of the character on the right?"
# image_file = "examples/breaking_bad.png"
# if debug:
#     image_file = f"videoDataProcess/{image_file}"

# args = type('Args', (), {
#     "model_path": model_path,
#     "model_base": None,
#     "model_name": get_model_name_from_path(model_path),
#     "query": prompt,
#     "conv_mode": None,
#     "image_file": image_file,
#     "sep": ",",
#     "temperature": 0,
#     "top_p": None,
#     "num_beams": 1,
#     "max_new_tokens": 512
# })()

# output=eval_model(args)
# print(output)


#!/usr/bin/python
# coding:utf8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_dense_optical_flow(prev_image, current_image):
    old_shape = current_image.shape
    prev_image_gray = cv2.cvtColor(prev_image,cv2.COLOR_BGR2GRAY)
    current_image_gray = cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
    assert current_image.shape == old_shape
    flow = None
    flow = cv2.calcOpticalFlowFarneback(prev=prev_image_gray,
                                        next=current_image_gray, flow=flow,
                                        pyr_scale=0.8, levels=15, winsize=5,
                                        iterations=10, poly_n=5, poly_sigma=0,
                                        flags=10)
    return flow

def compute_motion_score(img1, img2):
    flow = compute_dense_optical_flow(img1, img2)
    flow = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return np.mean(flow)

# 绘制光流
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


# img_path1 = "temporal/images/scene_0_begin.png"
# img_path2 = "temporal/images/scene_0_end.png"
img_path1 = "videoDataProcess/temporal/images/scene_19_begin.png"
img_path2 = "videoDataProcess/temporal/images/scene_19_middle.png"
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)
# 压缩图片
img1 = cv2.resize(img1, (320, 240))
img2 = cv2.resize(img2, (320, 240))
begin_time = time.time()
print(compute_motion_score(img1, img2))
end_time = time.time()
print(f"Time cost: {end_time - begin_time}s")

flow = compute_dense_optical_flow(img1, img2)
plt.imshow(draw_flow(img1, flow))
plt.savefig("flow.png")
# plt.savefig("videoDataProcess/flow.png")