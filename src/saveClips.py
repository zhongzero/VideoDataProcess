import os
import argparse
import imageio
import numpy as np
import torchvision
from scenedetect import detect, ContentDetector



def save_videos_grid(video, path, fps=8):
    # video: [T, C, H, W], value \in [0,255]
    outputs = []
    for img in video:
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        outputs.append(img)
    if os.path.dirname(path)!= "":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, macro_block_size = None)

    


def parse_args():
    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--video_path", type=str, default="data/世界首富的纨绔少爷/世界首富的纨绔少爷/1_no_sub.mp4")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--debug", help="use debug path", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    # args.debug = True
    video_path = args.video_path
    if args.debug:
        video_path = f"videoDataProcess/{video_path}"
        args.output_dir = f"videoDataProcess/{args.output_dir}"
    
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
    print("视频分辨率:", video_frames.size(2), video_frames.size(3))
    print("视频帧率:", info["video_fps"])
    
    for i, scene in enumerate(scene_list):
        begin_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        save_videos_grid(video_frames[begin_frame:end_frame], args.output_dir + f"/scene_{i}.mp4") # save as .mp4 or .gif

if __name__ == "__main__":
    main()