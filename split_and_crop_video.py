import os
import concurrent.futures
import cv2
import subprocess
import argparse
import face_alignment
from math import ceil

def extract_landmarks(frame):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    pred = fa.get_landmarks(frame)
    
    if pred is None or len(pred) == 0:
        return None
    
    pred = pred[0]
    min_x, min_y = pred[:, 0].min(), pred[:, 1].min()
    max_x, max_y = pred[:, 0].max(), pred[:, 1].max()

    width = max_x - min_x
    height = max_y - min_y

    res = min_x, min_y, width, height
    del fa

    return res

def crop_video(video_path, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the first frame
    ret, frame = cap.read()

    if not ret:
        print(f"Can't receive frame from {video_path}. Skipping...")
        cap.release()
        return None

    faces = extract_landmarks(frame)

    # Check if any face is detected
    if faces is not None:
        # Get the bounding box of the first face detected
        x, y, w, h = faces
        y = max(0, y - int(0.8*w))
        h = 3*h
        
        x = (2*x + w - h)//2
        w = h

        if w % 2 != 0:
            w += 1
            h += 1

        # Use ffmpeg to crop the video based on the bounding box
        command = f"ffmpeg -y -i \"{video_path}\" -filter:v \"crop={w}:{h}:{x}:{y}\" -b:v 4M \"{output_path}\""
        subprocess.call(command, shell=True)
        cap.release()
        return int(x), int(y), int(w), int(h)
    else:
        print(f"No face detected in {video_path}. Skipping crop...")
        cap.release()
        return None

def get_video_duration(input_file):
    """获取视频时长（秒）"""
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_file}"'
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return float(output)

def split_video_by_time(input_file, output_dir, segment_length=30, crop=False):
    """
    按时间间隔切分视频
    
    参数:
    - input_file: 输入视频文件路径
    - output_dir: 输出目录
    - segment_length: 每段视频的长度（秒）
    - crop: 是否对切片后的视频进行人脸裁剪
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    if crop:
        crop_dir = os.path.join(output_dir, "cropped")
        os.makedirs(crop_dir, exist_ok=True)
    
    # 获取视频时长
    duration = get_video_duration(input_file)
    
    # 计算需要切分的段数
    num_segments = ceil(duration / segment_length)
    
    # 文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    split_files = []
    
    for i in range(num_segments):
        start_time = i * segment_length
        split_output = os.path.join(output_dir, f"{base_name}_part{i+1:03d}.mp4")
        
        # 使用FFmpeg切分视频
        cmd = f'ffmpeg -i "{input_file}" -ss {start_time} -t {segment_length} -c:v libx264 -c:a aac -strict experimental -b:a 128k "{split_output}"'
        subprocess.call(cmd, shell=True)
        
        print(f"已创建切片: {split_output}")
        split_files.append(split_output)
    
    # 如果需要裁剪，则对每个切片进行裁剪
    if crop and split_files:
        crop_videos(split_files, crop_dir)
    
    return split_files

def split_video_by_count(input_file, output_dir, num_segments=10, crop=False):
    """
    将视频平均切分为指定数量的片段
    
    参数:
    - input_file: 输入视频文件路径
    - output_dir: 输出目录
    - num_segments: 要切分的片段数量
    - crop: 是否对切片后的视频进行人脸裁剪
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    if crop:
        crop_dir = os.path.join(output_dir, "cropped")
        os.makedirs(crop_dir, exist_ok=True)
    
    # 获取视频时长
    duration = get_video_duration(input_file)
    
    # 计算每段视频的长度
    segment_length = duration / num_segments
    
    # 文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    split_files = []
    
    for i in range(num_segments):
        start_time = i * segment_length
        split_output = os.path.join(output_dir, f"{base_name}_part{i+1:03d}.mp4")
        
        # 使用FFmpeg切分视频
        cmd = f'ffmpeg -i "{input_file}" -ss {start_time} -t {segment_length} -c:v libx264 -c:a aac -strict experimental -b:a 128k "{split_output}"'
        subprocess.call(cmd, shell=True)
        
        print(f"已创建切片: {split_output}")
        split_files.append(split_output)
    
    # 如果需要裁剪，则对每个切片进行裁剪
    if crop and split_files:
        crop_videos(split_files, crop_dir)
    
    return split_files

def crop_videos(video_paths, output_dir, n_processes=4):
    """
    对多个视频进行人脸裁剪处理
    
    参数:
    - video_paths: 视频文件路径列表
    - output_dir: 输出目录
    - n_processes: 并行处理的进程数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出路径
    output_paths = []
    for video_path in video_paths:
        base_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, base_name)
        output_paths.append(output_path)
    
    # 使用多进程处理视频裁剪
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        # 创建裁剪任务
        futures = [executor.submit(crop_video, video_path, output_path) 
                  for video_path, output_path in zip(video_paths, output_paths)]
        
        # 收集结果
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                print(f"已裁剪: {output_paths[i]}")
                results.append((output_paths[i], result))
            else:
                print(f"无法裁剪: {video_paths[i]}")
    
    # 保存裁剪信息
    if results:
        crop_info_path = os.path.join(output_dir, 'crop_info.txt')
        with open(crop_info_path, 'w') as file:
            for path, (x, y, w, h) in results:
                file.write(f"{path}: {x}, {y}, {w}, {h}\n")
        print(f"裁剪信息已保存到: {crop_info_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 MP4 视频切分成多个小视频并进行人脸裁剪')
    parser.add_argument('--input', required=True, help='输入视频文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--mode', choices=['time', 'count'], default='time', help='切分模式: time(按时间) 或 count(按数量)')
    parser.add_argument('--segment_length', type=int, default=30, help='每段视频的长度（秒）, 仅在 time 模式下使用')
    parser.add_argument('--num_segments', type=int, default=10, help='要切分的片段数量, 仅在 count 模式下使用')
    parser.add_argument('--crop', action='store_true', help='是否对切片后的视频进行人脸裁剪')
    parser.add_argument('--n_processes', type=int, default=4, help='并行处理的进程数（用于裁剪）')
    parser.add_argument('--only_crop', action='store_true', help='仅对输入目录中的视频进行裁剪，不进行切片')
    
    args = parser.parse_args()
    
    if args.only_crop:
        # 仅裁剪模式
        if os.path.isdir(args.input):
            video_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if video_files:
                crop_videos(video_files, args.output_dir, args.n_processes)
            else:
                print(f"在目录 {args.input} 中未找到视频文件")
        else:
            print(f"仅裁剪模式需要输入一个目录")
    else:
        # 切片并可选裁剪模式
        if args.mode == 'time':
            split_video_by_time(args.input, args.output_dir, args.segment_length, args.crop)
        else:
            split_video_by_count(args.input, args.output_dir, args.num_segments, args.crop) 