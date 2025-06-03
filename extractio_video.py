
'''
pip install pydub
sudo apt-get install ffmpeg

python split_audio_by_silence.py 输入音频文件.wav 输出目录 --silence_len 1000 --silence_thresh -40
'''
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import argparse

def split_audio(input_file, output_dir, silence_len=1000, silence_thresh=-40):
    # 加载音频文件
    audio = AudioSegment.from_file(input_file)
    # 按静默分割
    chunks = split_on_silence(
        audio,
        min_silence_len=silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200  # 保留部分静默
    )
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 保存分割后的音频
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(output_dir, f"{i:04d}.wav")
        chunk.export(out_path, format="wav")
        print(f"保存: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据静默分割音频")
    parser.add_argument("input_file", help="输入音频文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--silence_len", type=int, default=1000, help="静默时间长度(ms)")
    parser.add_argument("--silence_thresh", type=int, default=-40, help="静默阈值(dBFS)")
    args = parser.parse_args()
    split_audio(args.input_file, args.output_dir, args.silence_len, args.silence_thresh)
