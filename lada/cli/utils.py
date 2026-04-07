# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import mimetypes
import os
import pathlib
import subprocess
import sys
import time

from tqdm import tqdm
from wcwidth import wcswidth

from lada import ModelFiles
from lada.compute_targets import get_compute_targets
from lada.utils import VideoMetadata, video_utils

COL_SEP = "  "

def wcrjust(text, length, padding=' '):
    return text + padding * max(0, (length - wcswidth(text)))

def is_video_path(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        return False

    if sys.version_info >= (3, 13):
        mime_type, _ = mimetypes.guess_file_type(file_path)
    else:
        mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        return False
    return mime_type.lower().startswith("video/")


def list_video_files(directory_path: str, recursive: bool = False) -> list[str]:
    video_files: list[str] = []
    root = pathlib.Path(directory_path)
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if not is_video_path(str(path)):
            continue
        video_files.append(str(path))
    return sorted(video_files)


def build_output_file_path(
    input_file_path: str,
    output_directory: str,
    output_file_pattern: str,
    input_root_directory: str | None = None,
    preserve_relative_paths: bool = False,
) -> str:
    output_file_name = output_file_pattern.replace("{orig_file_name}", pathlib.Path(input_file_path).stem)
    if not preserve_relative_paths or input_root_directory is None:
        return os.path.join(output_directory, output_file_name)

    relative_parent = pathlib.Path(os.path.relpath(pathlib.Path(input_file_path).parent, input_root_directory))
    if str(relative_parent) == ".":
        return os.path.join(output_directory, output_file_name)
    return os.path.join(output_directory, str(relative_parent), output_file_name)

def setup_input_and_output_paths(input_arg, output_arg, output_file_pattern):
    single_file_input = os.path.isfile(input_arg)

    if single_file_input:
        input_files = [os.path.abspath(input_arg)]
    else:
        input_files = list_video_files(input_arg)

    if len(input_files) == 0:
        print(_("No video files found"))
        sys.exit(1)

    if single_file_input:
        if not output_arg:
            input_file_path = input_files[0]
            output_dir_path = str(pathlib.Path(input_file_path).parent)
            output_files = [build_output_file_path(input_file_path, output_dir_path, output_file_pattern)]
        elif os.path.isdir(output_arg):
            input_file_path = input_files[0]
            output_files = [build_output_file_path(input_file_path, output_arg, output_file_pattern)]
        else:
            output_files = [output_arg]
    else:
        if output_arg:
            if not os.path.exists(output_arg):
                os.makedirs(output_arg)
            output_dir_path = output_arg
        else:
            output_dir_path = str(pathlib.Path(input_files[0]).parent)
        output_files = [build_output_file_path(input_file_path, output_dir_path, output_file_pattern) for input_file_path in input_files]

    assert len(input_files) == len(output_files)

    return input_files, output_files

def _dump_table(table):
    row_count = len(table)
    col_count = len(table[0])
    col_widths = [0] * col_count
    for row in table:
        for col_i, col in enumerate(row):
            col_widths[col_i] = max(wcswidth(col), col_widths[col_i])
    s = ""
    for row_i, row in enumerate(table):
        for col_i, col in enumerate(row):
            s += f"{COL_SEP}{wcrjust(col, col_widths[col_i])}"
        if row_i < (row_count - 1):
            s += "\n"
        if row_i == 0:
            for col_i, col in enumerate(row):
                s += f"{COL_SEP}{col_widths[col_i] * "-"}"
            s += "\n"
    print(s)

def dump_encoders():
    from lada.utils.video_utils import get_video_encoder_codecs, get_human_readable_hardware_device_name
    encoders = get_video_encoder_codecs()
    print(_("Available video encoders:"))
    table = [[_("Name"), _("Description"), _("Hardware-accelerated"), _("Hardware devices")]]
    for e in encoders:
        hardware = _("Yes") if e.hardware_encoder else ""
        devices = ", ".join([get_human_readable_hardware_device_name(device) for device in e.hardware_devices])
        table.append([e.name, e.long_name, hardware, devices])
    _dump_table(table)

def dump_torch_devices():
    print(_("Available devices:"))
    table = [[_("Device"), _("Description"), _("Status"), _("Notes")]]
    for target in get_compute_targets(include_experimental=True):
        status = _("Available") if target.available else _("Unavailable")
        notes = target.notes if target.notes else (_("Experimental") if target.experimental else "")
        table.append([target.id, target.description, status, notes])
    _dump_table(table)

def dump_available_detection_models():
    modelfiles = ModelFiles.get_detection_models()
    print(_("Available detection models:"))
    if len(modelfiles) == 0:
        print(f"{COL_SEP}{_("None!")}")
    else:
        table = [[_("Name"), _("Description"), _("Path")]]
        for modelfile in modelfiles:
            table.append([modelfile.name, modelfile.description if modelfile.description else "", modelfile.path])
        _dump_table(table)

def dump_available_restoration_models():
    modelfiles = ModelFiles.get_restoration_models()
    print(_("Available restoration models:"))
    if len(modelfiles) == 0:
        print(f"{COL_SEP}{_("None!")}")
    else:
        table = [[_("Name"), _("Description"), _("Path")]]
        for modelfile in modelfiles:
            table.append([modelfile.name, modelfile.description if modelfile.description else "", modelfile.path])
        _dump_table(table)

def dump_available_encoding_presets():
    print(_("Available encoding presets:"))
    encoding_presets = video_utils.get_encoding_presets()
    if len(encoding_presets) == 0:
        s += f"\n{COL_SEP}{_("None!")}"
    else:
        table = [[_("Name"), _("Description")]]
        for preset in encoding_presets:
            table.append([preset.name, preset.description])
        _dump_table(table)

def dump_encoder_options(encoder: str):
    result = subprocess.run(["ffmpeg", "-loglevel", "quiet", "-h", f"encoder={encoder}"], capture_output=True, text=True)
    text = result.stdout.strip().replace("Exiting with exit code 0", "").strip()
    print(text)

class TranslatableHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
        super(TranslatableHelpFormatter, self).__init__(*args, **kwargs)

    def add_usage(self, usage, actions, groups, prefix=None):
        prefix = _("Usage: ")
        args = usage, actions, groups, prefix
        self._add_item(self._format_usage, args)

class Progressbar:
    def __init__(self, video_metadata: VideoMetadata):
        self.frame_processing_durations_buffer = []
        self.video_metadata = video_metadata
        processed_frame_budget = max(video_metadata.frames_count - 1, 0)
        self.frame_processing_durations_buffer_min_len = min(processed_frame_budget, int(video_metadata.video_fps * 15))
        self.frame_processing_durations_buffer_max_len = min(processed_frame_budget, int(video_metadata.video_fps * 120))
        self.error = False

        # Use {unit} instead of {postfix} as tqdm will add an additional comma without a way to overwrite this behavior (https://github.com/tqdm/tqdm/issues/712)
        BAR_FORMAT = _("Processing video: {done_percent}%|{bar}|Processed: {time_done} ({frames_done}f){bar_suffix}")
        BAR_FORMAT_TQDM = BAR_FORMAT.format(done_percent="{percentage:3.0f}", bar="{bar}", time_done="{elapsed}", frames_done="{n_fmt}", bar_suffix="{desc}")
        initial_estimating_bar_suffix = _(" | Remaining: ? | Speed: ?")
        self.tqdm = tqdm(dynamic_ncols=True, total=video_metadata.frames_count, bar_format=BAR_FORMAT_TQDM, desc=initial_estimating_bar_suffix)
        self.duration_start = None

    def init(self):
        self.duration_start = time.time()

    def close(self, ensure_completed_bar=False):
        if ensure_completed_bar:
            # On some media files the frame count, which is used to set up total of the progressbar, is not correct.
            # To prevent not showing not 100% completed bar update total to actual number of frames and refresh before closing
            if not self.error and self.tqdm.total != self.tqdm.n:
                self.tqdm.total = self.tqdm.n
                self._update_time_remaining_and_speed(completed=True)
                self.tqdm.refresh()
        self.tqdm.close()

    def update(self):
        duration_end = time.time()
        duration = duration_end - self.duration_start
        self.duration_start = duration_end

        if (
            self.frame_processing_durations_buffer_max_len > 0 and
            len(self.frame_processing_durations_buffer) >= self.frame_processing_durations_buffer_max_len
        ):
            self.frame_processing_durations_buffer.pop(0)
        self.frame_processing_durations_buffer.append(duration)

        self._update_time_remaining_and_speed()

        self.tqdm.update()

    def _get_mean_processing_duration(self):
        return sum(self.frame_processing_durations_buffer) / len(self.frame_processing_durations_buffer)

    def _format_duration(self, duration_s):
        if not duration_s or duration_s == -1:
            return "0:00"
        seconds = int(duration_s)
        minutes = int(seconds / 60)
        hours = int(minutes / 60)
        seconds = seconds % 60
        minutes = minutes % 60
        hours, minutes, seconds = int(hours), int(minutes), int(seconds)
        time = f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"
        return time

    def _update_time_remaining_and_speed(self, completed=False) -> float | None:
        frames_remaining = self.tqdm.format_dict['total'] - self.tqdm.format_dict['n'] if not completed else 0
        enough_datapoints =  len(self.frame_processing_durations_buffer) > self.frame_processing_durations_buffer_min_len
        if enough_datapoints:
            mean_duration = self._get_mean_processing_duration()
            time_remaining_s = frames_remaining * mean_duration
            time_remaining = self._format_duration(time_remaining_s)
            speed_fps = f"{1. / mean_duration:.1f}"
            self.tqdm.desc = _(" | Remaining: {time_remaining} ({frames_remaining}f) | Speed: {speed_fps}fps").format(time_remaining=time_remaining, frames_remaining=frames_remaining, speed_fps=speed_fps)
