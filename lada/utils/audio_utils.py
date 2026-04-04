# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging

import av
import io
import os
import subprocess
import shutil
from time import perf_counter
from typing import Optional
from lada.utils import video_utils, os_utils

logger = logging.getLogger(__name__)

def combine_audio_video_files(av_video_metadata: video_utils.VideoMetadata, tmp_v_video_input_path, av_video_output_path):
    start = perf_counter()
    profile: dict[str, str | bool | float | None] = {
        "audio_codec": None,
        "had_audio": False,
    }
    probe_started_at = perf_counter()
    audio_codec = get_audio_codec(av_video_metadata.video_file)
    profile["audio_probe_s"] = perf_counter() - probe_started_at
    profile["audio_codec"] = audio_codec
    if audio_codec:
        profile["had_audio"] = True
        needs_audio_reencoding = not is_output_container_compatible_with_input_audio_codec(audio_codec, av_video_output_path)
        needs_video_delay = av_video_metadata.start_pts > 0
        profile["needs_audio_reencoding"] = needs_audio_reencoding
        profile["needs_video_delay"] = needs_video_delay

        cmd = ["ffmpeg", "-y", "-loglevel", "quiet"]
        cmd += ["-i", av_video_metadata.video_file]
        if needs_video_delay > 0:
            delay_in_seconds = float(av_video_metadata.start_pts * av_video_metadata.time_base)
            cmd += ["-itsoffset", str(delay_in_seconds)]
        cmd += ["-i", tmp_v_video_input_path]
        if needs_audio_reencoding:
            cmd += ["-c:v", "copy"]
        else:
            cmd += ["-c", "copy"]
        cmd += ["-map", "1:v:0"]
        cmd += ["-map", "0:a:0"]
        cmd += [av_video_output_path]
        mux_started_at = perf_counter()
        subprocess.run(cmd, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
        profile["audio_mux_s"] = perf_counter() - mux_started_at
    else:
        copy_started_at = perf_counter()
        shutil.copy(tmp_v_video_input_path, av_video_output_path)
        profile["video_copy_s"] = perf_counter() - copy_started_at
    cleanup_started_at = perf_counter()
    os.remove(tmp_v_video_input_path)
    profile["cleanup_s"] = perf_counter() - cleanup_started_at
    profile["total_s"] = perf_counter() - start
    return profile

def get_audio_codec(file_path: str) -> Optional[str]:
    cmd = f"ffprobe -loglevel error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1"
    cmd = cmd.split() + [file_path]
    cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    audio_codec = cmd_result.stdout.decode('utf-8').strip().lower()
    return audio_codec if len(audio_codec) > 0 else None

def is_output_container_compatible_with_input_audio_codec(audio_codec: str, output_path: str) -> bool:
    file_extension = os.path.splitext(output_path)[1]
    file_extension = file_extension.lower()
    if file_extension in ('.mp4', '.m4v'):
        output_container_format = "mp4"
    elif file_extension == '.mkv':
        output_container_format = "matroska"
    else:
        logger.info(f"Couldn't determine video container format based on file extension: {file_extension}")
        return False

    buf = io.BytesIO()
    with av.open(buf, 'w', output_container_format) as container:
        return audio_codec in container.supported_codecs
