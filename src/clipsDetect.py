import os
import time
import argparse
from scenedetect import detect, ContentDetector
import torch
import torchvision.io
import cv2
import numpy as np
import matplotlib.pyplot as plt

from share4v.mm_utils import get_model_name_from_path
from share4v.model.builder import load_pretrained_model
# from share4v.eval.run_share4v import eval_model
from my_run_share4v import eval_model
from clipt_score import calc_clipt_score
import pyiqa




fir_caption_generation = True
tokenizer, model, image_processor, context_len = None, None, None, None

def caption_generation(img_path):
    begin_time = time.time()

    model_path = "Lin-Chen/ShareGPT4V-7B"
    prompt = "Provide a detailed yet concise description of the image that allows DALL-E or Midjourney to recreate it, focusing on key elements such as subjects, background, and objects. Describe colors, textures, lighting, and spatial relationships between objects, ensuring accuracy. If there is motion, vividly depict its direction and nature. Aim for a clear, comprehensive description, but avoid unnecessary length."
    image_file = img_path
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    global fir_caption_generation
    if fir_caption_generation:
        global tokenizer, model, image_processor, context_len
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, args.model_name)
        fir_caption_generation = False
    output=eval_model(args, tokenizer, model, image_processor)
    # 去除回车符
    output = output.replace("\n", " ")
    
    end_time = time.time()
    print(f"Time cost: {end_time - begin_time}s")
    
    return output

def allCaptionGeneration(video_path, scene_list, video_frames, temporal_dir, save_imgs=False):
    tmp_img_dir = os.path.join(temporal_dir, "tmp_images")
    os.makedirs(tmp_img_dir, exist_ok=True)
    
    debug_img_dir = os.path.join(temporal_dir, "images")
    os.makedirs(debug_img_dir, exist_ok=True)
    
    for i, scene in enumerate(scene_list):
        scene_info = f"Video path: {video_path}\nScene start frame: {scene[0].get_frames()}\nScene end frame: {scene[1].get_frames()-1}\n"
        if save_imgs:
            frame_begin = video_frames[scene[0].get_frames()]
            frame_end = video_frames[scene[1].get_frames() - 1]
            frame_middle = video_frames[(scene[0].get_frames() + scene[1].get_frames() - 1)//2]
            torchvision.io.write_png(frame_begin, f"{debug_img_dir}/scene_{i}_begin.png")
            torchvision.io.write_png(frame_end, f"{debug_img_dir}/scene_{i}_end.png")
            torchvision.io.write_png(frame_middle, f"{debug_img_dir}/scene_{i}_middle.png")
        
        # # 每个镜头的第一帧
        # frame_begin = video_frames[scene[0].get_frames()]
        # img_path = f"{tmp_img_dir}/scene_begin.png"
        # torchvision.io.write_png(frame_begin, img_path)
        # # caption generation
        # caption = caption_generation(img_path)
        # print(f"Caption of the begin frame in {i}-th scene: ", caption)
        # # save caption
        # scene_info = scene_info + f"Caption of the begin frame in {i}-th scene: {caption}\n"
        
        # # 每个镜头的最后一帧
        # frame_end = video_frames[scene[1].get_frames() - 1]
        # img_path = f"{tmp_img_dir}/scene_end.png"
        # torchvision.io.write_png(frame_end, img_path)
        # # caption generation
        # caption = caption_generation(img_path)
        # print(f"Caption of the end frame in {i}-th scene: ", caption)
        # # save caption
        # scene_info = scene_info + f"Caption of the end frame in {i}-th scene: {caption}\n"
        
        # 每个镜头的中间帧
        frame_middle = video_frames[(scene[0].get_frames() + scene[1].get_frames() - 1)//2]
        img_path = f"{tmp_img_dir}/scene_middle.png"
        torchvision.io.write_png(frame_middle, img_path)
        # caption generation
        caption = caption_generation(img_path)
        print(f"Caption of the middle frame in {i}-th scene: ", caption)
        # save caption
        # scene_info = scene_info + f"Caption of the middle frame in {i}-th scene: {caption}\n"
        scene_info = scene_info + f"Caption of the {i}-th scene: {caption}\n"
        
        with open(f"{temporal_dir}/scene_{i}_info.txt", 'w') as f:
            f.write(scene_info)

def qalign_score(input, qalign): # input: (N, 3, H, W), RGB, 0 ~ 1  , only support N = 1 currently
    quality_score = qalign(input, task_='quality')
    aesthetic_score = qalign(input, task_='aesthetic')
    return quality_score, aesthetic_score

def allQalignScoreCalculation(scene_list, video_frames, temporal_dir):
    # print(pyiqa.list_models())
    qalign = pyiqa.create_metric('qalign').cuda()
    for i, scene in enumerate(scene_list):
        scene_info = ""
        quality_scores = []
        aesthetic_scores = []
        
        # 每个镜头的第一帧
        frame_begin = video_frames[scene[0].get_frames()]
        # qalign score
        quality_score, aesthetic_score = qalign_score(frame_begin.unsqueeze(0) / 255.0, qalign)
        print("Qalign quality_score: ", quality_score)
        print("Qalign aesthetic_score: ", aesthetic_score)
        quality_scores.append(quality_score.cpu())
        aesthetic_scores.append(aesthetic_score.cpu())
        # # save qalign score
        # scene_info = scene_info + f"Qalign quality score of the begin frame in {i}-th scene: {quality_score.item()}\n"
        # scene_info = scene_info + f"Qalign aesthetic score of the begin frame in {i}-th scene: {aesthetic_score.item()}\n"
        
        # 每个镜头的最后一帧
        frame_end = video_frames[scene[1].get_frames() - 1]
        # qalign score
        quality_score, aesthetic_score = qalign_score(frame_end.unsqueeze(0) / 255.0, qalign)
        print("quality_score: ", quality_score)
        print("aesthetic_score: ", aesthetic_score)
        quality_scores.append(quality_score.cpu())
        aesthetic_scores.append(aesthetic_score.cpu())
        # # save qalign score
        # scene_info = scene_info + f"Qalign quality score of the end frame in {i}-th scene: {quality_score.item()}\n"
        # scene_info = scene_info + f"Qalign aesthetic score of the end frame in {i}-th scene: {aesthetic_score.item()}\n"
        
        # 每个镜头的中间帧
        frame_middle = video_frames[(scene[0].get_frames() + scene[1].get_frames() - 1)//2]
        # qalign score
        quality_score, aesthetic_score = qalign_score(frame_middle.unsqueeze(0) / 255.0, qalign)
        print("quality_score: ", quality_score)
        print("aesthetic_score: ", aesthetic_score)
        quality_scores.append(quality_score.cpu())
        aesthetic_scores.append(aesthetic_score.cpu())
        # # save qalign score
        # scene_info = scene_info + f"Qalign quality score of the middle frame in {i}-th scene: {quality_score.item()}\n"
        # scene_info = scene_info + f"Qalign aesthetic score of the middle frame in {i}-th scene: {aesthetic_score.item()}\n"
    
        scene_info = scene_info + f"Qalign quality score of the {i}-th scene: {np.mean(quality_scores).item()}\n"
        scene_info = scene_info + f"Qalign aesthetic score of the {i}-th scene: {np.mean(aesthetic_scores).item()}\n"
        
        with open(f"{temporal_dir}/scene_{i}_info.txt", 'a') as f:
            f.write(scene_info)

def allClipTScoreCalculation(scene_list, video_frames, temporal_dir):
    scene_caption = []
    for i in range(len(scene_list)):
        with open(f"{temporal_dir}/scene_{i}_info.txt", 'r') as f:
            file = f.read()
            # begin_caption = file.split(f"Caption of the begin frame in {i}-th scene: ")[1].split("\n")[0]
            # end_caption = file.split(f"Caption of the end frame in {i}-th scene: ")[1].split("\n")[0]
            # middle_caption = file.split(f"Caption of the middle frame in {i}-th scene: ")[1].split("\n")[0]
            caption = file.split(f"Caption of the {i}-th scene: ")[1].split("\n")[0]
            scene_caption.append(caption) # 使用中间帧的caption作为镜头的caption

    tmp_img_dir = os.path.join(temporal_dir, "tmp_images")
    os.makedirs(tmp_img_dir, exist_ok=True)
    for i, scene in enumerate(scene_list):
        # 每个镜头的第一帧，最后一帧，中间帧
        frame_begin = video_frames[scene[0].get_frames()]
        frame_end = video_frames[scene[1].get_frames() - 1]
        frame_middle = video_frames[(scene[0].get_frames() + scene[1].get_frames() - 1)//2]
        for file in os.listdir(tmp_img_dir):
            os.remove(os.path.join(tmp_img_dir, file)) # 清空文件夹
        torchvision.io.write_png(frame_begin, f"{tmp_img_dir}/scene_begin.png")
        torchvision.io.write_png(frame_end, f"{tmp_img_dir}/scene_end.png")
        torchvision.io.write_png(frame_middle, f"{tmp_img_dir}/scene_middle.png")
        # clip-t score
        clipt_score = calc_clipt_score(scene_caption[i], tmp_img_dir)
        print(f"scene_caption[{i}]: {scene_caption[i]}")
        print(f"Clip-t score: {clipt_score}")
        # save clip score
        scene_info = f"Clip-t score of the {i}-th scene: {clipt_score}\n"
        with open(f"{temporal_dir}/scene_{i}_info.txt", 'a') as f:
            f.write(scene_info)

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

def allMotionScoreCalculation(scene_list, video_frames, temporal_dir, delta_frame_num=10, image_compress = False, image_size=(320, 240)):
    # delta_frame_num: 间隔帧数
    for i, scene in enumerate(scene_list):
        begin_frame_id=scene[0].get_frames()
        end_frame_id=scene[1].get_frames() - 1
        motion_scores = []
        begin_time = time.time()
        for j in range(begin_frame_id, end_frame_id - delta_frame_num + 1, delta_frame_num):
            frame_before = video_frames[j].permute(1, 2, 0).numpy()
            frame_after = video_frames[j+delta_frame_num].permute(1, 2, 0).numpy()
            if image_compress:
                frame_before = cv2.resize(frame_before, image_size)
                frame_after = cv2.resize(frame_after, image_size)
            motion_score = compute_motion_score(frame_before, frame_after)
            motion_scores.append(motion_score)
        end_time = time.time()
        print(f"Time cost: {end_time - begin_time}s")
        if len(motion_scores) == 0:
            motion_scores.append(-1) # 镜头太短直接舍去
        print(f"Motion scores of the {i}-th scene: {np.mean(motion_scores)}")
        
        if image_compress:
            scene_info = f"(Compress) Motion scores of the {i}-th scene: {np.mean(motion_scores)}\n"
        else :
            scene_info = f"Motion scores of the {i}-th scene: {np.mean(motion_scores)}\n"
        with open(f"{temporal_dir}/scene_{i}_info.txt", 'a') as f:
            f.write(scene_info)

def allResultIntegration(scene_list, temporal_dir, output_path):
    with open(output_path, 'a') as f:
        for i in range(len(scene_list)):
            with open(f"{temporal_dir}/scene_{i}_info.txt", 'r') as f2:
                file = f2.read()
                f.write(file)
        

def videoProcess(args):

    video_path = args.video_path
    if args.debug:
        video_path = f"videoDataProcess/{video_path}"
    
    time1 = time.time()
    
    # 分镜
    scene_list = detect(video_path, ContentDetector())
    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            i+1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),))
    
    
    # 读取视频文件
    video_frames, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')
    video_frames = video_frames.permute(0, 3, 1, 2)   # [T, H, W, C] -> [T, C, H, W]

    # 查看视频信息
    print("视频帧数:", video_frames.size(0))
    print("视频分辨率:", video_frames.size(1), video_frames.size(2))
    print("视频帧率:", info["video_fps"])
    
    temporal_dir = args.temporal_dir
    if args.debug:
        temporal_dir = f"videoDataProcess/{temporal_dir}"
    os.makedirs(temporal_dir, exist_ok=True)
    
    output_path = args.output_path
    if args.debug:
        output_path = f"videoDataProcess/{output_path}"
    
    time2=time.time()
    allCaptionGeneration(video_path, scene_list, video_frames, temporal_dir, save_imgs=True)
    time3=time.time()
    allQalignScoreCalculation(scene_list, video_frames, temporal_dir)
    time4=time.time()
    allClipTScoreCalculation(scene_list, video_frames, temporal_dir)
    time5=time.time()
    allMotionScoreCalculation(scene_list, video_frames, temporal_dir, image_compress=False)
    time6=time.time()
    allMotionScoreCalculation(scene_list, video_frames, temporal_dir, image_compress=True)
    time7=time.time()
    allResultIntegration(scene_list, temporal_dir, output_path)
    
    print(f"Read video total time cost: {time2 - time1}s")
    print(f"Caption genration total time cost: {time3 - time2}s")
    print(f"Qalign score calculation total time cost: {time4 - time3}s")
    print(f"Clip-t score calculation total time cost: {time5 - time4}s")
    print(f"Motion score calculation total time cost: {time6 - time5}s")
    print(f"(Compress) Motion score calculation total time cost: {time7 - time6}s")
    print(f"Total time cost: {time7 - time1}s")

def parse_args():
    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--video_path", type=str, default="data/世界首富的纨绔少爷/世界首富的纨绔少爷/1.mp4")
    parser.add_argument("--temporal_dir", type=str, default="temporal")
    parser.add_argument("--output_path", type=str, default="results.txt")
    parser.add_argument("--debug", help="use debug path", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    videoProcess(args)

if __name__ == "__main__":
    main()