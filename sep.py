import os
from pathlib import Path

import av
import av.error
import demucs
import demucs.api
import numpy as np
import torch
from torch import Tensor

repo = Path("models/sep/Demucs_Models/v3_v4_repo")
models = demucs.api.list_models(repo)
model = "htdemucs_6s"
assert model in models["single"] or model in models["bag"]
sep = demucs.api.Separator(
    model=model,
    repo=repo,
    device="cuda:0",
    # segment=44,
)
separate_tensor = sep.separate_tensor


# --------------------------------------------------------------------------
# 步骤 2: 将任意音频文件转换为函数所需的 Tensor
# --------------------------------------------------------------------------
def load_audio_to_tensor(file_path: str) -> tuple[Tensor, int]:
    """
    使用 PyAV 加载音频文件并转换为torch.Tensor。

    Returns:
        A tuple (waveform_tensor, sample_rate).
        The tensor is float32 and has shape (channels, samples).
    """
    try:
        container = av.open(file_path)
    except av.error.FFmpegError as e:
        print(f"无法打开文件: {file_path}. 错误: {e}")
        raise

    # 选择第一个音频流
    stream = container.streams.audio[0]

    # 获取采样率
    sample_rate = stream.rate

    # 解码所有帧并拼接
    frames = []
    for frame in container.decode(stream):
        # to_ndarray() 返回一个 (channels, samples) 的 numpy 数组
        frames.append(frame.to_ndarray())

    if not frames:
        raise ValueError("音频文件中没有找到有效的音频帧。")

    # 将所有帧拼接成一个大的 numpy 数组
    waveform_np = np.concatenate(frames, axis=1)

    # 确保数据类型为 float32
    # PyAV 可能返回 int16, int32, float32, float64 等
    if waveform_np.dtype != np.float32:
        # 如果是整数类型, 需要归一化到 [-1.0, 1.0]
        if np.issubdtype(waveform_np.dtype, np.integer):
            max_val = np.iinfo(waveform_np.dtype).max
            waveform_np = waveform_np.astype(np.float32) / max_val
        else:
            # 其他浮点类型直接转换
            waveform_np = waveform_np.astype(np.float32)

    # 从 numpy 数组创建 torch.Tensor
    waveform_tensor = torch.from_numpy(waveform_np)

    return waveform_tensor, sample_rate


# --------------------------------------------------------------------------
# 步骤 3: 将分离出的音轨 Tensor 保存为 WAV 文件
# --------------------------------------------------------------------------
def save_tensors_to_wav(
    stems: dict[str, Tensor], sample_rate: int, output_dir: str, original_filename: str
) -> None:
    """
    使用 PyAV 将音轨字典中的每个 Tensor 保存为 WAV 文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    for stem_name, stem_tensor in stems.items():
        output_filename = f"{original_filename}_{stem_name}.wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"正在保存音轨 '{stem_name}' 到 '{output_path}'...")

        # 确保 Tensor 在 CPU 上并且是 contiguous 的
        stem_tensor = stem_tensor.cpu().contiguous()

        # 将 float32 Tensor 转换回 int16 (WAV 文件常用格式)
        # 1.裁剪到 [-1.0, 1.0] 范围, 防止溢出
        stem_tensor = torch.clamp(stem_tensor, -1.0, 1.0)
        # 2.乘以 int16 的最大值并转换类型
        stem_np_int16 = (stem_tensor.numpy() * 32767).astype(np.int16)

        # 使用 PyAV 保存
        num_channels = stem_np_int16.shape[0]
        layout = "stereo" if num_channels == 2 else "mono"

        with av.open(output_path, mode="w") as out_container:
            out_stream = out_container.add_stream(
                "pcm_s16le",  # 16-bit little-endian PCM, WAV 标准格式
                rate=sample_rate,
                layout=layout,
            )

            # 将 numpy 数组包装成 AudioFrame
            frame = av.AudioFrame.from_ndarray(
                stem_np_int16, format="s16p", layout=layout
            )
            frame.sample_rate = sample_rate

            # 编码并写入文件
            for packet in out_stream.encode(frame):
                out_container.mux(packet)

            # Flush a-v stream
            for packet in out_stream.encode(None):
                out_container.mux(packet)


def main() -> None:
    # --- 准备一个用于测试的音频文件 ---
    # 如果你已经有一个音频文件, 比如 "my_song.mp3", 可以直接使用它。

    INPUT_FILE = "samples/countdown_to_zero_luotianyi.mp3"  # 你可以改成你的文件名, 如 'song.flac', 'voice.ogg' 等
    OUTPUT_DIR = "tmp/"

    if not os.path.exists(INPUT_FILE):
        print(f"未找到输入文件 '{INPUT_FILE}'")
        return

    # 1. 加载音频文件到 Tensor
    print("\n[步骤 1/3] 正在加载音频并转换为 Tensor...")
    try:
        input_wav, input_sr = load_audio_to_tensor(INPUT_FILE)
        print(f"音频加载成功: shape={input_wav.shape}, sr={input_sr}")
    except (ValueError, av.error.FFmpegError) as e:
        print(f"音频加载失败: {e}")
        return  # 退出程序

    # 2. 调用音轨分离函数
    print("\n[步骤 2/3] 正在调用音轨分离函数...")
    # 假设模型返回的音轨采样率与模型内部采样率一致, 我们用 44100
    _, separated_stems = separate_tensor(wav=input_wav, sr=input_sr)
    MODEL_INTERNAL_SR = sep.samplerate  # 假设我们知道模型输出是这个采样率

    # 3. 将分离出的音轨 Tensor 保存为 WAV 文件
    print("\n[步骤 3/3] 正在将分离出的音轨保存为 WAV 文件...")
    filename_without_ext = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    save_tensors_to_wav(
        stems=separated_stems,
        sample_rate=MODEL_INTERNAL_SR,  # 使用模型输出的采样率
        output_dir=OUTPUT_DIR,
        original_filename=filename_without_ext,
    )

    print("\n处理完成！所有音轨已保存到 '{}' 文件夹中。".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
