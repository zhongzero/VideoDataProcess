import argparse

def mark_low_score(list_of_clips, score_name, threshold_rate): #标记score最小的一批clip，score名称为score_name， 比例为threshold_rate
    scores = [clip[score_name] for clip in list_of_clips]
    scores.sort()
    threshold = scores[int(len(scores) * threshold_rate)]
    for clip in list_of_clips:
        if clip[score_name] < threshold:
            clip["Remove"] = True
    

# Video path: data/世界首富的纨绔少爷/世界首富的纨绔少爷/1.mp4
# Scene start frame: 0
# Scene end frame: 31
# Caption of the 0-th scene: The image captures a moment of tranquility and adventure. A helicopter, painted in a sleek black color, is seen soaring above a serene lake. The helicopter is equipped with a large rotor on top and a smaller one on the tail, both spinning rapidly as they cut through the air.  The helicopter is positioned in the top left corner of the image, flying from left to right. It's slightly tilted downwards, suggesting it's in the process of landing. The backdrop is a picturesque scene of a lake, its calm waters reflecting the surrounding trees. The trees, adorned with autumn foliage, add a splash of color to the scene.  The sky above is painted in hues of orange, indicating that the photo was taken at sunset. The warm glow of the setting sun casts long shadows and bathes the entire scene in a soft, golden light. The image beautifully encapsulates the thrill of helicopter travel against the backdrop of nature's tranquility.
# Qalign quality score of the 0-th scene: 3.611328125
# Qalign aesthetic score of the 0-th scene: 2.92578125
# Clip-t score of the 0-th scene: 0.125
# Motion scores of the 0-th scene: 115.67241668701172
# (Compress) Motion scores of the 0-th scene: 30.06256675720215

def filterData(args, use_compress_motion_score):
    list_of_clips = []
    with open(args.input_file_path, "r") as f:
        lines = f.readlines()
        # print("Number of lines in file: ", len(lines))
        clip = {}
        for line in lines:
            if line.startswith("Video path: "):
                clip = {}
                video_path = line.split("Video path: ")[1][:-1]
                clip["video_path"] = video_path
            elif line.startswith("Scene start frame: "):
                scene_start_frame = int(line.split("Scene start frame: ")[1][:-1])
                clip["scene_start_frame"] = scene_start_frame
            elif line.startswith("Scene end frame: "):
                scene_end_frame = int(line.split("Scene end frame: ")[1][:-1])
                clip["scene_end_frame"] = scene_end_frame
            elif line.startswith("Caption of the "):
                caption = line.split("Caption of the ")[1].split("-th scene: ")[1][:-1]
                id = int(line.split("Caption of the ")[1].split("-th scene: ")[0])
                clip["caption"] = caption
                clip["id"] = id
            elif line.startswith("Qalign quality score of the "):
                qalign_quality_score = float(line.split("Qalign quality score of the ")[1].split("-th scene: ")[1][:-1])
                clip["qalign_quality_score"] = qalign_quality_score
            elif line.startswith("Qalign aesthetic score of the "):
                qalign_aesthetic_score = float(line.split("Qalign aesthetic score of the ")[1].split("-th scene: ")[1][:-1])
                clip["qalign_aesthetic_score"] = qalign_aesthetic_score
            elif line.startswith("Clip-t score of the "):
                clip_t_score = float(line.split("Clip-t score of the ")[1].split("-th scene: ")[1][:-1])
                clip["clip_t_score"] = clip_t_score
            elif line.startswith("Motion scores of the "):
                motion_score = float(line.split("Motion scores of the ")[1].split("-th scene: ")[1][:-1])
                clip["motion_score"] = motion_score
            elif line.startswith("(Compress) Motion scores of the "):
                compress_motion_score = float(line.split("(Compress) Motion scores of the ")[1].split("-th scene: ")[1][:-1])
                clip["compress_motion_score"] = compress_motion_score
                list_of_clips.append(clip)
    print("Number of clips: ", len(list_of_clips))
    for clip in list_of_clips:
        clip["Remove"] = False
        # print(clip)
    
    # mark_low_score(list_of_clips,score_name = "qalign_quality_score", threshold_rate = 0.2)
    # mark_low_score(list_of_clips,score_name = "qalign_aesthetic_score", threshold_rate = 0.2)
    # mark_low_score(list_of_clips,score_name = "clip_t_score", threshold_rate = 0.2)
    if use_compress_motion_score==False:
        mark_low_score(list_of_clips,score_name = "motion_score", threshold_rate = 0.4)
    else:
        mark_low_score(list_of_clips,score_name = "compress_motion_score", threshold_rate = 0.4)
    
    remove_num = 0
    for clip in list_of_clips:
        if clip["Remove"]:
            remove_num += 1
    print("Number of clips to remove: ", remove_num)
    
    with open(args.output_file_path, "w") as f:
        for clip in list_of_clips:
            if clip["Remove"]:
                continue
            f.write("Video path: " + clip["video_path"] + "\n")
            f.write("Scene start frame: " + str(clip["scene_start_frame"]) + "\n")
            f.write("Scene end frame: " + str(clip["scene_end_frame"]) + "\n")
            f.write("Caption of the " + str(clip["id"]) + "-th scene: " + clip["caption"] + "\n")
            f.write("Qalign quality score of the " + str(clip["id"]) + "-th scene: " + str(clip["qalign_quality_score"]) + "\n")
            f.write("Qalign aesthetic score of the " + str(clip["id"]) + "-th scene: " + str(clip["qalign_aesthetic_score"]) + "\n")
            f.write("Clip-t score of the " + str(clip["id"]) + "-th scene: " + str(clip["clip_t_score"]) + "\n")
            f.write("Motion scores of the " + str(clip["id"]) + "-th scene: " + str(clip["motion_score"]) + "\n")
            f.write("(Compress) Motion scores of the " + str(clip["id"]) + "-th scene: " + str(clip["compress_motion_score"]) + "\n")
    
    return list_of_clips
            

def parse_args():
    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--input_file_path", type=str, default="results.txt")
    parser.add_argument("--output_file_path", type=str, default="results_filtered.txt")
    parser.add_argument("--debug", help="use debug path", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    list_of_clips1=filterData(args, use_compress_motion_score= False)
    list_of_clips2=filterData(args, use_compress_motion_score= True)
    
    # 比较两个list_of_clips有多少clip的Remove不一样
    num_diff=0
    for clip1, clip2 in zip(list_of_clips1, list_of_clips2):
        if clip1["Remove"] != clip2["Remove"]:
            num_diff+=1
    print("Number of clips with different Remove: ", num_diff)

if __name__ == "__main__":
    main()