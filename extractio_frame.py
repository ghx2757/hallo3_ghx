'''
python extract_frame.py input.mp4 1 ./output      # 提取第一帧
python extract_frame.py input.mp4 -2 ./output     # 提取倒数第二帧
'''

import os
import subprocess
import sys

def get_total_frames(video_path):
    # 获取视频总帧数
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return int(result.stdout.strip())
    except:
        # 某些编码下nb_read_frames不可用，退而求其次
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return int(result.stdout.strip())

def extract_frame(video_path, frame_id, output_dir):
    total_frames = get_total_frames(video_path)
    if frame_id == 0 or abs(frame_id) > total_frames:
        print(f"帧号{frame_id}超出范围，总帧数为{total_frames}")
        return
    # 负数帧号处理
    if frame_id < 0:
        frame_id = total_frames + frame_id + 1
    # ffmpeg帧号从0开始，-vf "select=eq(n\,frame_id-1)"
    select_expr = f"eq(n\\,{frame_id-1})"
    output_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
    cmd = [
        "ffmpeg", "-i", video_path, "-vf", f"select='{select_expr}'", "-vframes", "1", output_path, "-y"
    ]
    subprocess.run(cmd)
    print(f"已保存: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python extract_frame.py <视频路径> <帧号> <输出目录>")
        print("帧号支持负数，如-1为倒数第一帧")
        sys.exit(1)
    video_path = sys.argv[1]
    frame_id = int(sys.argv[2])
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    extract_frame(video_path, frame_id, output_dir)
