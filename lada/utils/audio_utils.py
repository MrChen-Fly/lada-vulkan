# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging

import av
import io
import os
import subprocess
import shutil
import tempfile
from time import perf_counter
from typing import Optional
from lada.utils import video_utils, os_utils

logger = logging.getLogger(__name__)

_MP4_COMPATIBLE_AUDIO_CODECS = {"aac", "mp4a", "alac", "ac3", "eac3"}
_WEBM_COMPATIBLE_AUDIO_CODECS = {"opus", "vorbis"}


def _get_output_container_format(output_path: str) -> Optional[str]:
    file_extension = os.path.splitext(output_path)[1].lower()
    if file_extension in (".mp4", ".m4v"):
        return "mp4"
    if file_extension == ".webm":
        return "webm"
    if file_extension == ".mkv":
        return "matroska"
    return None


def is_output_container_compatible_with_input_codec(codec_name: str, output_path: str) -> bool:
    output_container_format = _get_output_container_format(output_path)
    if output_container_format is None:
        file_extension = os.path.splitext(output_path)[1].lower()
        logger.info(f"Couldn't determine video container format based on file extension: {file_extension}")
        return False

    buf = io.BytesIO()
    with av.open(buf, 'w', output_container_format) as container:
        return codec_name in container.supported_codecs


def _get_audio_reencode_settings(audio_codec: str, output_path: str) -> tuple[bool, list[str], Optional[str]]:
    output_container_format = _get_output_container_format(output_path)
    if output_container_format == "mp4":
        normalized_codec = audio_codec.lower()
        if normalized_codec not in _MP4_COMPATIBLE_AUDIO_CODECS:
            return True, ["-c:a", "aac", "-b:a", "192k"], "aac"
        return False, [], normalized_codec
    if output_container_format == "webm":
        normalized_codec = audio_codec.lower()
        if normalized_codec not in _WEBM_COMPATIBLE_AUDIO_CODECS:
            return True, ["-c:a", "libopus", "-b:a", "192k"], "opus"
        return False, [], normalized_codec
    return (not is_output_container_compatible_with_input_audio_codec(audio_codec, output_path), [], None)


def can_copy_video_file_to_output(
    video_codec: str | None,
    audio_codec: str | None,
    output_path: str,
) -> tuple[bool, str | None]:
    output_container_format = _get_output_container_format(output_path)
    if output_container_format is None:
        return False, f"unsupported output container '{os.path.splitext(output_path)[1].lower()}'"

    if not video_codec:
        return False, "missing input video codec"
    if not is_output_container_compatible_with_input_codec(video_codec.lower(), output_path):
        return False, f"video codec '{video_codec}' is not compatible with {output_container_format}"

    if audio_codec and not is_output_container_compatible_with_input_audio_codec(audio_codec.lower(), output_path):
        return False, f"audio codec '{audio_codec}' is not compatible with {output_container_format}"
    return True, None


def copy_or_remux_video_file(source_video_path: str, output_path: str) -> dict[str, str | float]:
    start = perf_counter()
    profile: dict[str, str | float] = {
        "mode": "copy",
        "source_video_path": source_video_path,
    }
    source_extension = os.path.splitext(source_video_path)[1].lower()
    output_extension = os.path.splitext(output_path)[1].lower()

    if source_extension == output_extension:
        copy_started_at = perf_counter()
        shutil.copy(source_video_path, output_path)
        profile["copy_s"] = perf_counter() - copy_started_at
        profile["total_s"] = perf_counter() - start
        return profile

    profile["mode"] = "remux"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", source_video_path, "-map", "0", "-c", "copy"]
    if _get_output_container_format(output_path) == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [output_path]

    remux_started_at = perf_counter()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=os_utils.get_subprocess_startup_info(),
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Failed to remux '{source_video_path}' to '{output_path}'"
            + (f": {stderr}" if stderr else ".")
        )

    profile["remux_s"] = perf_counter() - remux_started_at
    profile["total_s"] = perf_counter() - start
    return profile


def _stabilize_mux_video_input_path(tmp_v_video_input_path: str) -> str | None:
    # On Windows, ffmpeg can occasionally misread a freshly closed WebM temp
    # file unless the file is first materialized through an OS-level copy pass.
    if os.name != "nt" or os.path.splitext(tmp_v_video_input_path)[1].lower() != ".webm":
        return None

    tmp_dir = os.path.dirname(tmp_v_video_input_path) or None
    tmp_stem = os.path.splitext(os.path.basename(tmp_v_video_input_path))[0]
    tmp_ext = os.path.splitext(tmp_v_video_input_path)[1]
    fd, shadow_copy_path = tempfile.mkstemp(
        prefix=f"{tmp_stem}.muxshadow.",
        suffix=tmp_ext,
        dir=tmp_dir,
    )
    os.close(fd)
    shutil.copy2(tmp_v_video_input_path, shadow_copy_path)
    return shadow_copy_path


def combine_audio_video_files(av_video_metadata: video_utils.VideoMetadata, tmp_v_video_input_path, av_video_output_path):
    start = perf_counter()
    profile: dict[str, str | bool | float | None] = {
        "audio_codec": None,
        "had_audio": False,
        "output_audio_codec": None,
        "stabilized_video_input": False,
    }
    probe_started_at = perf_counter()
    audio_codec = get_audio_codec(av_video_metadata.video_file)
    profile["audio_probe_s"] = perf_counter() - probe_started_at
    profile["audio_codec"] = audio_codec
    shadow_video_input_path = None
    try:
        if audio_codec:
            profile["had_audio"] = True
            needs_audio_reencoding, audio_reencode_args, output_audio_codec = _get_audio_reencode_settings(
                audio_codec,
                av_video_output_path,
            )
            needs_video_delay = av_video_metadata.start_pts > 0
            profile["needs_audio_reencoding"] = needs_audio_reencoding
            profile["needs_video_delay"] = needs_video_delay
            profile["output_audio_codec"] = output_audio_codec if needs_audio_reencoding else audio_codec

            shadow_video_input_path = _stabilize_mux_video_input_path(tmp_v_video_input_path)
            profile["stabilized_video_input"] = shadow_video_input_path is not None

            cmd = ["ffmpeg", "-y", "-loglevel", "quiet"]
            cmd += ["-i", av_video_metadata.video_file]
            if needs_video_delay > 0:
                delay_in_seconds = float(av_video_metadata.start_pts * av_video_metadata.time_base)
                cmd += ["-itsoffset", str(delay_in_seconds)]
            cmd += ["-i", tmp_v_video_input_path]
            if needs_audio_reencoding:
                cmd += ["-c:v", "copy"]
                cmd += audio_reencode_args
            else:
                cmd += ["-c", "copy"]
            cmd += ["-map", "1:v:0"]
            cmd += ["-map", "0:a:0"]
            cmd += [av_video_output_path]
            mux_started_at = perf_counter()
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=os_utils.get_subprocess_startup_info(),
            )
            profile["audio_mux_s"] = perf_counter() - mux_started_at
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Failed to mux audio into '{av_video_output_path}'"
                    + (f": {stderr}" if stderr else ".")
                )
        else:
            copy_started_at = perf_counter()
            shutil.copy(tmp_v_video_input_path, av_video_output_path)
            profile["video_copy_s"] = perf_counter() - copy_started_at
    finally:
        cleanup_started_at = perf_counter()
        cleanup_paths = [tmp_v_video_input_path]
        if shadow_video_input_path:
            cleanup_paths.append(shadow_video_input_path)
        for cleanup_path in cleanup_paths:
            if cleanup_path and os.path.exists(cleanup_path):
                os.remove(cleanup_path)
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
    return is_output_container_compatible_with_input_codec(audio_codec, output_path)
