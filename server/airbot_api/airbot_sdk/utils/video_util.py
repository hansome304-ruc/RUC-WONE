import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class H264VideoWriter:
    """H264视频写入器基类"""

    def __init__(
        self,
        filename: str,
        width: int,
        height: int,
        fps: int,
        pixel_format: str = "rgb24",
    ):
        """
        初始化H264视频写入器

        参数:
            filename: 输出文件名 (.mp4)
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            pixel_format: 像素格式 (默认: rgb24)
        """
        self.filename = str(Path(filename).with_suffix(".mp4"))
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format

        self.command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            pixel_format,
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-tune",
            "zerolatency",
            "-g",
            str(fps),
            "-keyint_min",
            str(fps),
            "-bf",
            "0",
            "-flags",
            "+cgop",
            "-x264-params",
            "nal-hrd=cbr:force-cfr=1",
            # "-b:v",
            # "5M",
            # "-minrate",
            # "5M",
            # "-maxrate",
            # "5M",
            # "-bufsize",
            # "2M",
            "-metadata",
            f"creation_time={datetime.now().isoformat()}",
            self.filename,
        ]

        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        """
        写入一帧数据

        参数:
            frame: numpy.ndarray，与指定像素格式匹配的图像数据
        """
        try:
            self.process.stdin.write(frame.tobytes())
        except IOError:
            logger.exception("写入视频帧时发生错误")
            raise

    def release(self):
        """关闭写入器并完成视频文件"""
        if self.process:
            try:
                self.process.stdin.close()
                stderr = self.process.stderr.read()
                self.process.wait(timeout=5)
                if self.process.returncode != 0:
                    logger.info(f"FFmpeg错误: {stderr.decode()}")
            except Exception:
                logger.exception("关闭视频写入器时发生错误")
            finally:
                self.process = None


class DepthVideoWriter:
    def __init__(self, filename, width, height, fps=30):
        """
        初始化16位深度视频写入器

        参数:
            filename: 输出文件名 (.mkv)
            width: 视频宽度
            height: 视频高度
            fps: 帧率
        """
        self.filename = str(Path(filename).with_suffix(".mkv"))
        self.width = width
        self.height = height
        self.fps = fps

        # FFV1
        self.command = [
            "ffmpeg",
            "-y",  # 覆盖已存在的文件
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "gray16le",  # 16位灰度格式
            "-r",
            str(fps),
            "-i",
            "-",  # 从管道读入
            "-c:v",
            "ffv1",
            "-level",
            "3",  # FFV1版本3
            "-coder",
            "1",  # 范围编码器
            "-context",
            "1",
            "-g",
            "1",  # 每帧都是关键帧
            "-threads",
            "12",
            "-slices",
            "24",  # 并行编码的切片数
            self.filename,
        ]

        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        """
        写入一帧16位深度数据

        参数:
            frame: numpy.ndarray，类型为uint16的深度图
        """
        assert frame.dtype == np.uint16, "输入帧必须是16位无符号整数类型"
        assert frame.shape == (
            self.height,
            self.width,
        ), f"帧大小必须为 {self.width}x{self.height}"

        # 确保数据是小端字节序（FFmpeg期望的格式）
        frame = frame.astype("<u2")
        self.process.stdin.write(frame.tobytes())

    def release(self):
        """关闭写入器并完成视频文件"""
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            stderr = self.process.stderr.read()
            if self.process.returncode != 0:
                raise RuntimeError(f"FFmpeg错误: {stderr.decode()}")
            self.process = None


class DepthVideoReader:
    def __init__(self, filename):
        """
        初始化16位深度视频读取器

        参数:
            filename: 输入文件名 (.mkv)
        """
        self.filename = filename
        self.frame_count = 0  # 帧计数器
        self.start_time = None  # 起始时间
        self.current_timestamp = None  # 当前帧时间戳

        # 使用FFmpeg获取视频信息
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        import json

        info = json.loads(probe.stdout)
        stream = info["streams"][0]

        self.width = int(stream["width"])
        self.height = int(stream["height"])
        fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
        self.fps = fps_num / fps_den
        self.frame_duration = 1.0 / self.fps  # 每帧持续时间（秒）

        # 获取视频时长信息
        duration_probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        duration_info = json.loads(duration_probe.stdout)
        try:
            self.duration = float(duration_info["format"]["duration"])
        except (KeyError, ValueError):
            self.duration = None

        # 设置基本命令
        self.command = [
            "ffmpeg",
            "-i",
            filename,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-v",
            "error",
            "-",
        ]

        self.process = None
        self.frame_size = self.width * self.height * 2  # 16位 = 2字节

    def read(self):
        """
        读取一帧数据

        返回:
            (success, frame): success为布尔值，frame为numpy数组
        """
        if self.process is None:
            self.process = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.frame_count = 0
            self.start_time = time.time()
            self.current_timestamp = 0.0

        raw_frame = self.process.stdout.read(self.frame_size)
        if len(raw_frame) < self.frame_size:
            self.release()
            return False, None

        frame = np.frombuffer(raw_frame, dtype="<u2").reshape(self.height, self.width)

        # 更新帧计数和时间戳
        self.frame_count += 1
        self.current_timestamp = (self.frame_count - 1) * self.frame_duration

        return True, frame

    def read_with_timestamp(self):
        """
        读取一帧数据并返回时间戳

        返回:
            (success, timestamp, frame):
                success: 布尔值，表示是否成功读取
                timestamp: 相对于视频开始的时间戳（秒）
                frame: numpy数组，表示帧数据
        """
        success, frame = self.read()
        if not success:
            return False, None, None

        return True, self.current_timestamp, frame

    def get_frame_info(self):
        """
        获取当前帧的信息

        返回:
            dict: 包含帧号和相对时间戳的字典
        """
        return {
            "frame_number": self.frame_count,
            "timestamp": self.current_timestamp,  # 相对于视频开始的时间（秒）
        }

    def get_progress(self):
        """
        获取视频读取进度

        返回:
            float: 0.0-1.0之间的进度值，如果无法确定则返回None
        """
        if self.duration is not None and self.current_timestamp is not None:
            return min(1.0, self.current_timestamp / self.duration)
        return None

    def seek_to_frame(self, frame_number):
        """
        尝试跳转到指定帧号（注意：这是一个近似实现，可能不精确）

        参数:
            frame_number: 目标帧号

        返回:
            bool: 是否成功跳转
        """
        if self.process is not None:
            self.release()

        if frame_number <= 0:
            # 重新开始读取
            self.process = None
            return True

        # 计算目标时间点（秒）
        target_time = (frame_number - 1) * self.frame_duration

        # 使用-ss参数跳转到指定时间点
        seek_command = [
            "ffmpeg",
            "-ss",
            str(target_time),
            "-i",
            self.filename,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-v",
            "error",
            "-",
        ]

        self.process = subprocess.Popen(
            seek_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # 更新帧计数和时间戳
        self.frame_count = frame_number
        self.current_timestamp = target_time
        self.start_time = time.time() - target_time

        return True

    def release(self):
        """关闭读取器"""
        if self.process:
            self.process.stdout.close()
            self.process.wait()
            self.process = None


class VideoProbe:
    """视频文件探测器"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.info = self._probe_video()
        self.summary_info = None

    def _probe_video(self) -> Dict:
        """获取视频详细信息"""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-count_frames",
            "-count_packets",
            str(self.video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError:
            logger.exception("Error probing video")
            return {}
        except json.JSONDecodeError:
            logger.exception("Error parsing ffprobe output")
            return {}

    def get_encoding_info(self):
        """获取视频的编码信息"""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            self.video_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            info = json.loads(result.stdout)
            stream = info["streams"][0]

            # logger.info(stream)

            encoding = {
                "vcodec": stream.get("codec_name"),
                "pix_fmt": stream.get("pix_fmt"),
                "level": stream.get("level"),
                "profile": stream.get("profile"),
                "bit_rate": stream.get("bit_rate"),
            }

            return encoding
        except Exception:
            logger.exception("Error getting encoding info")
            return None

    def _get_video_stream(self) -> Optional[Dict]:
        """获取视频流信息"""
        if not self.info:
            return None

        for stream in self.info.get("streams", []):
            if stream.get("codec_type") == "video":
                return stream
        return None

    def get_frame_count(self) -> int:
        """获取总帧数"""
        stream = self._get_video_stream()
        if not stream:
            return 0

        # 优先使用nb_read_frames
        frame_count = stream.get("nb_read_frames")
        if frame_count:
            return int(frame_count)

        # 备选方案：使用duration和帧率计算
        duration = float(stream.get("duration", 0))
        fps = self.get_fps()
        if duration and fps:
            return int(duration * fps)

        return 0

    def get_fps(self) -> float:
        """获取实际帧率"""
        stream = self._get_video_stream()
        if not stream:
            return 0.0

        # 优先使用avg_frame_rate
        avg_rate = stream.get("avg_frame_rate", "")
        if avg_rate and avg_rate != "0/0":
            num, den = map(int, avg_rate.split("/"))
            return num / den if den != 0 else 0.0

        # 备选：使用r_frame_rate
        r_rate = stream.get("r_frame_rate", "")
        if r_rate and r_rate != "0/0":
            num, den = map(int, r_rate.split("/"))
            return num / den if den != 0 else 0.0

        return 0.0

    def get_duration(self) -> float:
        """获取视频时长（秒）"""
        if not self.info:
            return 0.0

        # 优先使用format中的duration
        duration = self.info.get("format", {}).get("duration")
        if duration:
            return float(duration)

        # 备选：使用视频流中的duration
        stream = self._get_video_stream()
        if stream and stream.get("duration"):
            return float(stream["duration"])

        return 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """获取视频分辨率"""
        stream = self._get_video_stream()
        if not stream:
            return (0, 0)

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        return (width, height)

    def get_codec(self) -> str:
        """获取视频编码格式"""
        stream = self._get_video_stream()
        if not stream:
            return "unknown"

        return stream.get("codec_name", "unknown")

    def get_pixel_format(self) -> str:
        """获取像素格式"""
        stream = self._get_video_stream()
        if not stream:
            return "unknown"

        return stream.get("pix_fmt", "unknown")

    def get_bitrate(self) -> int:
        """获取视频比特率"""
        if not self.info:
            return 0

        # 优先使用format中的bit_rate
        bitrate = self.info.get("format", {}).get("bit_rate")
        if bitrate:
            return int(bitrate)

        # 备选：使用视频流中的bit_rate
        stream = self._get_video_stream()
        if stream and stream.get("bit_rate"):
            return int(stream["bit_rate"])

        return 0

    def get_file_size(self) -> Tuple[int, str]:
        """获取文件大小（返回原始字节数和格式化大小）"""
        size_bytes = self.video_path.stat().st_size
        return size_bytes, self.format_size(size_bytes)

    def get_compression_ratio(self) -> float:
        """计算视频压缩率（仅对RGB视频有意义）"""
        width, height = self.get_resolution()
        frame_count = self.get_frame_count()

        # 假设RGB24位色深
        raw_size = width * height * 3 * frame_count  # 3 bytes per pixel
        actual_size = self.video_path.stat().st_size

        if raw_size == 0:
            return 0.0

        return actual_size / raw_size

    def get_summary_info(self, show_detail: bool = True):
        """获取视频信息"""
        if self.summary_info is None:
            width, height = self.get_resolution()
            duration = self.get_duration()
            frame_count = self.get_frame_count()
            fps = self.get_fps()
            summary_info = {
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
                "frame_count": frame_count,
            }
            if show_detail:
                bitrate = self.get_bitrate()
                size_bytes, size_formatted = self.get_file_size()
                summary_info.update(
                    {
                        "bitrate": bitrate,
                        "size_bytes": size_bytes,
                        "size_formatted": size_formatted,
                    }
                )
            logger.info(f"summary_info {summary_info}")
            self.summary_info = summary_info
        return self.summary_info

    def print_summary(self):
        """打印视频摘要信息"""
        width, height = self.get_resolution()
        duration = self.get_duration()
        frame_count = self.get_frame_count()
        fps = self.get_fps()
        bitrate = self.get_bitrate()
        size_bytes, size_formatted = self.get_file_size()

        logger.info(f"\n视频文件: {self.video_path.name}")
        logger.info(f"文件大小: {size_formatted} ({size_bytes:,} 字节)")
        logger.info(f"分辨率: {width}x{height}")
        logger.info(f"时长: {duration:.2f}秒")
        logger.info(f"总帧数: {frame_count:,}")
        logger.info(f"帧率: {fps:.2f} fps")
        logger.info(f"编码格式: {self.get_codec()}")
        logger.info(f"像素格式: {self.get_pixel_format()}")
        logger.info(f"比特率: {bitrate / 1000:.2f} kbps")

        # 如果是RGB视频，显示压缩率
        if "rgb" in self.video_path.name.lower():
            compression_ratio = self.get_compression_ratio()
            logger.info(f"压缩率: {compression_ratio:.2%}")

    def get_storage_efficiency(self) -> Dict:
        """计算存储效率指标"""
        size_bytes, size_formatted = self.get_file_size()
        duration = self.get_duration()
        frame_count = self.get_frame_count()
        width, height = self.get_resolution()

        return {
            "size_per_second": size_bytes / duration if duration > 0 else 0,
            "size_per_frame": size_bytes / frame_count if frame_count > 0 else 0,
            "bits_per_pixel": (
                (size_bytes * 8) / (width * height * frame_count)
                if (width * height * frame_count) > 0
                else 0
            ),
        }

    @staticmethod
    def analyze_recording_session(session_path: str):
        """分析录制会话的存储情况"""
        session_dir = Path(session_path)

        # 查找所有视频文件
        rgb_files = list(session_dir.glob("*.mp4"))
        depth_files = list(session_dir.glob("*.mkv"))

        total_size = 0
        session_stats = {
            "rgb_videos": [],
            "depth_videos": [],
            "total_duration": 0,
            "total_frames": 0,
        }

        logger.info(f"\n分析录制会话: {session_dir.name}")
        logger.info(f"找到 {len(rgb_files)} 个RGB视频和 {len(depth_files)} 个深度视频")

        # 分析RGB视频
        logger.info("\nRGB视频分析:")
        for rgb_file in sorted(rgb_files):
            probe = VideoProbe(rgb_file)
            size_bytes, size_formatted = probe.get_file_size()
            total_size += size_bytes

            efficiency = probe.get_storage_efficiency()
            session_stats["rgb_videos"].append(
                {
                    "name": rgb_file.name,
                    "size": size_bytes,
                    "duration": probe.get_duration(),
                    "frames": probe.get_frame_count(),
                    "efficiency": efficiency,
                }
            )

            # logger.info(f"\n文件: {rgb_file.name}")
            # logger.info(f"大小: {size_formatted}")
            probe.print_summary()
            logger.info(f"每秒存储: {probe.format_size(int(efficiency['size_per_second']))}/s")
            logger.info(f"每帧大小: {probe.format_size(int(efficiency['size_per_frame']))}/frame")
            logger.info(f"像素位深: {efficiency['bits_per_pixel']:.2f} bits/pixel")

        # 分析深度视频
        logger.info("\n深度视频分析:")
        for depth_file in sorted(depth_files):
            probe = VideoProbe(depth_file)
            size_bytes, size_formatted = probe.get_file_size()
            total_size += size_bytes

            efficiency = probe.get_storage_efficiency()
            session_stats["depth_videos"].append(
                {
                    "name": depth_file.name,
                    "size": size_bytes,
                    "duration": probe.get_duration(),
                    "frames": probe.get_frame_count(),
                    "efficiency": efficiency,
                }
            )

            # logger.info(f"\n文件: {depth_file.name}")
            # logger.info(f"大小: {size_formatted}")
            probe.print_summary()
            logger.info(f"每秒存储: {probe.format_size(int(efficiency['size_per_second']))}/s")
            logger.info(f"每帧大小: {probe.format_size(int(efficiency['size_per_frame']))}/frame")
            logger.info(f"像素位深: {efficiency['bits_per_pixel']:.2f} bits/pixel")

        # 会话总结
        logger.info("\n会话总结:")
        logger.info(f"总存储大小: {VideoProbe.format_size(total_size)} ({total_size:,} 字节)")

        # 计算平均值
        if session_stats["rgb_videos"]:
            avg_rgb_size_per_frame = sum(v["size"] for v in session_stats["rgb_videos"]) / sum(
                v["frames"] for v in session_stats["rgb_videos"]
            )
            logger.info(
                f"RGB平均帧大小: {VideoProbe.format_size(int(avg_rgb_size_per_frame))}/frame"
            )

        if session_stats["depth_videos"]:
            avg_depth_size_per_frame = sum(v["size"] for v in session_stats["depth_videos"]) / sum(
                v["frames"] for v in session_stats["depth_videos"]
            )
            logger.info(
                f"深度平均帧大小: {VideoProbe.format_size(int(avg_depth_size_per_frame))}/frame"
            )

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """将字节数转换为人类可读格式"""
        if size_bytes == 0:
            return "0 B"

        size_names = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)

        return f"{s} {size_names[i]}"

    @staticmethod
    def verify_sync(video_paths: List[str]) -> bool:
        """验证多个视频的同步性

        Args:
            video_paths: 要验证的视频文件路径列表

        Returns:
            bool: 所有视频是否同步
        """
        if not video_paths or len(video_paths) < 2:
            logger.info("需要至少2个视频文件进行同步验证")
            return False

        probes = []
        frame_counts = []
        durations = []

        # 收集所有视频的信息
        logger.info("\n同步验证结果:")
        for path in video_paths:
            probe = VideoProbe(path)
            frames = probe.get_frame_count()
            duration = probe.get_duration()

            probes.append(probe)
            frame_counts.append(frames)
            durations.append(duration)

            logger.info(f"\n文件: {Path(path).name}")
            logger.info(f"帧数: {frames:,}")
            logger.info(f"时长: {duration:.2f}秒")
            logger.info(f"帧率: {probe.get_fps():.2f} fps")

        # 计算最大差异
        max_frame_diff = max(frame_counts) - min(frame_counts)
        max_duration_diff = max(durations) - min(durations)

        logger.info("\n差异分析:")
        logger.info(f"最大帧数差异: {max_frame_diff:,} 帧")
        logger.info(f"最大时长差异: {max_duration_diff:.3f} 秒")

        # 详细的两两比较
        logger.info("\n详细比较:")
        for i in range(len(video_paths)):
            for j in range(i + 1, len(video_paths)):
                frame_diff = abs(frame_counts[i] - frame_counts[j])
                duration_diff = abs(durations[i] - durations[j])

                file1 = Path(video_paths[i]).name
                file2 = Path(video_paths[j]).name

                logger.info(f"\n{file1} vs {file2}:")
                logger.info(f"帧数差异: {frame_diff:,} 帧")
                logger.info(f"时长差异: {duration_diff:.3f} 秒")

        # 判断是否同步（允许1帧的误差且时长差异小于0.1秒）
        is_synced = max_frame_diff <= 1 and max_duration_diff < 0.1

        logger.info(f"\n同步状态: {'正常' if is_synced else '不同步'}")

        # 如果不同步，给出可能的原因
        if not is_synced:
            logger.info("\n可能的问题:")
            if max_frame_diff > 1:
                logger.info("- 帧数不匹配，可能存在丢帧")
            if max_duration_diff >= 0.1:
                logger.info("- 时长差异过大，可能存在时间戳偏移")

            # 检查帧率是否一致
            fps_values = [probe.get_fps() for probe in probes]
            max_fps_diff = max(fps_values) - min(fps_values)
            if max_fps_diff > 0.1:  # 允许0.1fps的误差
                logger.info("- 帧率不一致，可能影响同步")

        return is_synced


def test_depth_video():
    """测试深度视频的写入和读取"""
    # 测试参数
    width, height = 640, 480
    fps = 30
    n_frames = 10
    test_file = "test_depth.mkv"

    # 生成测试数据
    logger.info("生成测试数据...")
    test_frames = [
        np.random.randint(0, 65536, (height, width), dtype=np.uint16) for _ in range(n_frames)
    ]

    # 写入测试数据
    logger.info("写入深度视频...")
    writer = DepthVideoWriter(test_file, width, height, fps)
    for frame in test_frames:
        writer.write(frame)
    writer.release()

    # 读取并验证数据
    logger.info("读取深度视频并验证...")
    reader = DepthVideoReader(test_file)
    read_frames = []

    while True:
        ret, frame = reader.read()
        if not ret:
            break
        read_frames.append(frame)

    reader.release()

    # 验证结果
    if len(test_frames) != len(read_frames):
        logger.info(f"帧数不匹配: 原始 {len(test_frames)} vs 读取 {len(read_frames)}")
        return False

    for i, (original, read) in enumerate(zip(test_frames, read_frames)):
        logger.info(f"{original.shape} {original.ndim} {original.dtype} {np.max(original)}")
        if not np.array_equal(original, read):
            logger.info(f"帧 {i} 数据不匹配")
            diff = np.abs(original.astype(np.int32) - read.astype(np.int32))
            logger.info(f"最大差异: {np.max(diff)}")
            logger.info(f"差异统计: {np.unique(diff, return_counts=True)}")
            return False

    logger.info("测试通过！所有数据完全匹配")
    os.remove(test_file)
    return True


def validate_local_data(mkv_path="tmp_task/episode_3/videos/test_realsense_depth.mkv"):
    width = 1280
    height = 720
    raw_frames = np.fromfile(
        f"{mkv_path}.npy",
        dtype=np.uint16,
    )

    frame_size = width * height
    n_frames = len(raw_frames) // frame_size

    # 重塑数组
    depth_frames = raw_frames.reshape((n_frames, height, width))
    logger.info(f"{depth_frames.shape} {len(depth_frames)}")

    reader = DepthVideoReader(mkv_path)
    read_frames = []

    while True:
        ret, frame = reader.read()
        if not ret:
            break
        read_frames.append(frame)

    reader.release()
    logger.info(len(read_frames))

    passed = True
    if len(depth_frames) != len(read_frames):
        passed = False
        logger.info("长度不匹配")
    for i, (original, read) in enumerate(zip(depth_frames, read_frames)):
        if not np.array_equal(original, read):
            passed = False
            logger.info(f"帧 {i} 数据不匹配")
            logger.info(f"{original.shape} {original.ndim} {original.dtype} {np.max(original)}")
            diff = np.abs(original.astype(np.int32) - read.astype(np.int32))
            logger.info(f"最大差异: {np.max(diff)}")
            logger.info(f"差异统计: {np.unique(diff, return_counts=True)}")

    if passed:
        logger.info("测试通过！所有数据完全匹配")


def convert_bgr_to_rgb_video(input_file, output_file=None):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    if output_file is None:
        output_file = input_file

    is_same_file = os.path.abspath(input_file) == os.path.abspath(output_file)
    temp_output = (
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        if is_same_file
        else output_file
    )

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise RuntimeError("无法打开输入视频文件")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-loglevel",
            "error",
            temp_output,
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg 处理失败: {process.stderr.read().decode()}")

        if is_same_file:
            if os.path.exists(output_file):
                os.remove(output_file)
            shutil.move(temp_output, output_file)

    except Exception as e:
        if is_same_file and os.path.exists(temp_output):
            os.remove(temp_output)
        raise e

    finally:
        cap.release()
        if "process" in locals():
            try:
                process.terminate()
            except Exception:
                pass


# 运行测试
if __name__ == "__main__":
    test_depth_video()

    # convert_bgr_to_rgb_video("tmp_task/episode_1/videos/test_2_realsense_rgb.mp4")

    # p = VideoProbe(
    #     "/media/huangjunwen/JunwenH's Drive/30k/UR5_2/2025_03_10/pick_up/episode_36/videos/global_realsense_depth.mkv"
    # )
    # logger.info(p.get_encoding_info())
    # validate_local_data("tmp_task/episode_1/videos/test_realsense_depth.mkv")
    # reader = DepthVideoReader("/data/tools/tmp_task/episode_1/videos/test_2_realsense_depth.mkv")
    # read_frames = []

    # frame_count = 0
    # while True:
    #     ret, frame = reader.read()
    #     if not ret:
    #         break
    #     frame_count += 1
    # logger.info(frame_count)
