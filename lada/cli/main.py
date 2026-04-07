# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
from time import perf_counter

# When frozen (PyInstaller) on macOS, multiprocessing spawn re-executes this
# executable with -B -S -I -c "from multiprocessing.resource_tracker import main; main(...)".
# PyTorch triggers this. Stdlib freeze_support() only handles worker processes
# (argv[1] == '--multiprocessing-fork'); it does not handle the resource_tracker
# process (which uses -c). See: Lib/multiprocessing/spawn.py `is_forking()` and
# `freeze_support()` in CPython.
# (https://github.com/python/cpython/blob/629a363ddd2889f023d5925506e61f5b6647accd/Lib/multiprocessing/spawn.py#L57-L80).
# Run the -c code and exit so argparse never sees it.
if getattr(sys, "frozen", False) and sys.platform == "darwin":
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-c" and i + 1 < len(sys.argv):
            exec(compile(sys.argv[i + 1], "<string>", "exec"))
            sys.exit(0)
        if sys.argv[i] == "-c":
            break

_CLI_PROCESS_START = perf_counter()

try:
    import torch
except ModuleNotFoundError:
    from lada import IS_FLATPAK
    if IS_FLATPAK:
        print(_("No GPU Add-On installed"))
        print(_("In order to use the application you need to install one of Lada's Flatpak Add-Ons that is compatible with your hardware"))
        sys.exit(1)
    else:
        raise

from lada import VERSION, ModelFiles
from lada.cli import batch_runner
from lada.cli import utils
from lada.compute_targets import (
    default_fp16_enabled_for_compute_target,
    describe_compute_target_issue,
    get_compute_target,
    get_default_compute_target,
    normalize_compute_target_id,
    resolve_torch_device,
)
from lada.utils import audio_utils, video_utils
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline import load_models
from lada.restorationpipeline.runtime_profiling import WallClockProfiler
from lada.utils.threading_utils import STOP_MARKER, ErrorMarker
from lada.utils.video_utils import get_video_meta_data, VideoWriter, get_default_preset_name


def _video_metadata_to_report(video_metadata) -> dict[str, object]:
    return {
        "video_file": video_metadata.video_file,
        "video_width": video_metadata.video_width,
        "video_height": video_metadata.video_height,
        "video_fps": float(video_metadata.video_fps),
        "video_fps_exact": str(video_metadata.video_fps_exact),
        "frames_count": video_metadata.frames_count,
        "duration": float(video_metadata.duration),
        "time_base": str(video_metadata.time_base),
        "start_pts": video_metadata.start_pts,
        "codec_name": video_metadata.codec_name,
    }


def _compute_target_to_report(compute_target) -> dict[str, object]:
    return {
        "id": compute_target.id,
        "description": compute_target.description,
        "runtime": compute_target.runtime,
        "available": compute_target.available,
        "torch_device": compute_target.torch_device,
        "notes": compute_target.notes,
        "experimental": compute_target.experimental,
    }


def _write_timing_report(report_path: str, report: dict[str, object]) -> None:
    pathlib.Path(report_path).parent.mkdir(exist_ok=True, parents=True)
    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)


def _is_full_video_passthrough(frame_restorer_profile: dict[str, object]) -> bool:
    mosaic_detector_profile = frame_restorer_profile.get("mosaic_detector")
    if not isinstance(mosaic_detector_profile, dict):
        return False
    return (
        int(frame_restorer_profile.get("clips_restored", 0)) == 0
        and int(frame_restorer_profile.get("frames_blended", 0)) == 0
        and int(mosaic_detector_profile.get("clips_emitted", 0)) == 0
    )


def _cleanup_intermediate_outputs(
    *,
    video_tmp_file_output_path: str,
    working_output_path: str | None,
    output_path: str,
) -> None:
    if os.path.exists(video_tmp_file_output_path):
        os.remove(video_tmp_file_output_path)
    if (
        working_output_path
        and os.path.exists(working_output_path)
        and os.path.normcase(working_output_path) != os.path.normcase(output_path)
    ):
        os.remove(working_output_path)


def setup_argparser() -> argparse.ArgumentParser:
    examples_header_text = _("Examples:")

    example1_text = _("Restore video with default settings:")
    example1_command = _("%(prog)s --input input.mp4")

    example2_text = _("Restore all videos found in the specified directory and save them to a different folder:")
    example2_command = _("%(prog)s --input path/to/input/dir/ --output /path/to/output/dir/")

    example3_text = _("Use Nvidia hardware-accelerated encoder by selecting a preset:")
    example3_command = _("%(prog)s --input input.mp4 --encoding-preset hevc-nvidia-gpu-hq")

    example4_text = _("Set encoding parameters directly without using an encoding preset:")
    example4_command = _("%(prog)s --input input.mp4 --encoder libx265 --encoder-options '-crf 26 -preset fast -x265-params log_level=error'")

    parser = argparse.ArgumentParser(
        usage=_('%(prog)s [options]'),
        description=_("Restore pixelated adult videos (JAV)"),
        epilog=_(textwrap.dedent(f'''\
            {examples_header_text}
                * {example1_text}
                    {example1_command}
                * {example2_text}
                     {example2_command}
                * {example3_text}
                    {example3_command}
                * {example4_text}
                    {example4_command}
            ''')),
        formatter_class=utils.TranslatableHelpFormatter,
        add_help=False)

    group_general = parser.add_argument_group(_('General'))
    group_general.add_argument('--input', type=str, help=_('Path to pixelated video file or directory containing video files'))
    group_general.add_argument('--output', type=str, help=_('Path used to save output file(s). If path is a directory then file name will be chosen automatically (see --output-file-pattern). If no output path was given then the directory of the input file will be used'))
    group_general.add_argument('--temporary-directory', type=str, default=tempfile.gettempdir(), help=_('Directory for temporary video files during restoration process. Alternatively, you can use the environment variable TMPDIR. (default: %(default)s)'))
    group_general.add_argument('--output-file-pattern', type=str, default="{orig_file_name}.restored.mp4", help=_("Pattern used to determine output file name(s). Used when input is a directory, or a file but no output path was specified. Must include the placeholder '{orig_file_name}'. (default: %(default)s)"))
    group_general.add_argument('--device', type=str, default=get_default_compute_target(), help=_('Device used for running Restoration and Detection models. Use "--list-devices" to see what\'s available (default: %(default)s)'))
    group_general.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=None, help=_("Reduces VRAM usage and may increase speed on supported GPUs, with negligible quality difference. Default: auto"))
    group_general.add_argument('--timing-report-path', type=str, default=None, help=_('Optional path used to write a JSON timing report for the full CLI pipeline'))
    group_general.add_argument('--list-devices', action='store_true', help=_("List available devices and exit"))
    group_general.add_argument('--version', action='store_true', help=_("Display version and exit"))
    group_general.add_argument('--help', action='store_true', help=_("Show this help message and exit"))

    batch_group = parser.add_argument_group(_('Batch Processing'))
    batch_group.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=False, help=_('Recursively scan nested directories when the input is a directory. (default: %(default)s)'))
    batch_group.add_argument('--preserve-relative-paths', action=argparse.BooleanOptionalAction, default=False, help=_('Preserve the input directory structure under the output directory. (default: %(default)s)'))
    batch_group.add_argument('--backup-root', type=str, default=None, help=_('Optional directory used to store source video backups before conversion'))
    batch_group.add_argument('--batch-state-path', type=str, default=None, help=_('Optional JSON file used to persist batch progress and failed-file retry order'))
    batch_group.add_argument('--force-backup', action=argparse.BooleanOptionalAction, default=False, help=_('Rewrite source backups even if an identical backup already exists. (default: %(default)s)'))
    batch_group.add_argument('--force-reconvert', action=argparse.BooleanOptionalAction, default=False, help=_('Re-run conversion even if the final output file already exists. (default: %(default)s)'))
    batch_group.add_argument('--retry-count', type=int, default=0, help=_('Number of additional retries for a failed file in batch mode. (default: %(default)s)'))
    batch_group.add_argument('--retry-delay-seconds', type=int, default=3, help=_('Delay between failed-file retries in batch mode. (default: %(default)s)'))
    batch_group.add_argument('--retry-failed-first', action=argparse.BooleanOptionalAction, default=False, help=_('When resuming a batch, retry previously failed files before untouched files. (default: %(default)s)'))
    batch_group.add_argument('--working-output-extension', type=str, default=None, help=_('Temporary encoded output extension used before moving to the final output path, for example ".mp4"'))
    batch_group.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=False, help=_('Show planned batch actions without writing backups or outputs. (default: %(default)s)'))
    batch_group.add_argument('--max-files', type=int, default=0, help=_('Limit the number of directory inputs processed in batch mode. 0 keeps all files. (default: %(default)s)'))

    export = parser.add_argument_group(_('Export'))
    export.add_argument('--encoding-preset', type=str, default=get_default_preset_name(), help=_('Select encoding preset by name. Use "--list-encoding-presets" to see what\'s available. Ignored if "--encoder" and "--encoder-options" are used (default: %(default)s)'))
    export.add_argument('--list-encoding-presets', action='store_true', help=_("List available encoding presets and exit"))
    export.add_argument('--encoder', type=str, help=_('Select video encoder by name. Use "--list-encoders" to see what\'s available. (default: %(default)s)'))
    export.add_argument('--list-encoders', action='store_true', help=_("List available encoders and exit"))
    export.add_argument('--encoder-options', type=str, help=_("Space-separated list of options for the encoder set via \"--encoder\". Use \"--list-encoder-options\" to see what's available. (default: %(default)s)"))
    export.add_argument('--list-encoder-options', metavar='ENCODER', type=str, help=_("List available options of the given encoder and exit"))
    export.add_argument('--mp4-fast-start',  default=False, action=argparse.BooleanOptionalAction, help=_("Allows playing the file while it's being written. Sets .mp4 mov flags \"frag_keyframe+empty_moov+faststart\". (default: %(default)s)"))

    group_restoration = parser.add_argument_group(_('Mosaic Restoration'))
    group_restoration.add_argument('--list-mosaic-restoration-models', action='store_true', help=_("List available restoration models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default='basicvsrpp-v1.2', help=_('Name of detection model or path to model weights file. Use "--list-mosaic-restoration-models" to see what\'s available. (default: %(default)s)'))
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str, default=None, help=_("Path to restoration model configuration file. You'll not have to set this unless you're training your own custom models"))
    group_restoration.add_argument('--max-clip-length', type=int, default=180, help=_('Maximum number of frames for restoration. Higher values improve temporal stability. Lower values reduce memory footprint. If set too low flickering could appear (default: %(default)s)'))

    group_detection = parser.add_argument_group(_('Mosaic Detection'))
    group_detection.add_argument('--mosaic-detection-model', type=str, default='v4-fast', help=_('Name of detection model or path to model weights file. Use "--list-mosaic-detection-models" to see what\'s available. (default: %(default)s)'))
    group_detection.add_argument('--list-mosaic-detection-models', action='store_true', help=_("List available detection models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_detection.add_argument('--detect-face-mosaics', action=argparse.BooleanOptionalAction, default=False, help=_("Detect and ignore areas of pixelated faces. Can prevent restoration artifacts but may worsen detection of NSFW mosaics. Available for models v3 and newer. (default: %(default)s)"))

    return parser

def process_video_file(input_path: str, output_path: str, temp_dir_path: str, device: torch.device, mosaic_restoration_model, mosaic_detection_model,
                       mosaic_restoration_model_name, preferred_pad_mode, max_clip_length, encoder: str, encoder_options: str, mp4_fast_start,
                       working_output_path: str | None = None):
    profiler = WallClockProfiler()
    started_at = perf_counter()
    encoded_output_path = working_output_path if working_output_path else output_path
    with profiler.measure("video_metadata_probe_s"):
        video_metadata = get_video_meta_data(input_path)

    with profiler.measure("frame_restorer_init_s"):
        frame_restorer = FrameRestorer(device, input_path, max_clip_length, mosaic_restoration_model_name,
                     mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode)
    success = True
    frames_written = 0
    audio_profile: dict[str, object] = {}
    frame_restorer_profile: dict[str, object] = {}
    video_writer: VideoWriter | None = None
    error_message: str | None = None
    video_tmp_file_output_path = os.path.join(temp_dir_path, f"{os.path.basename(os.path.splitext(encoded_output_path)[0])}.tmp{os.path.splitext(encoded_output_path)[1]}")
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    if working_output_path:
        pathlib.Path(encoded_output_path).parent.mkdir(exist_ok=True, parents=True)
    frame_restorer_progressbar = utils.Progressbar(video_metadata)
    try:
        with profiler.measure("frame_restorer_start_s"):
            frame_restorer.start()
        frame_restorer_progressbar.init()
        with profiler.measure("video_writer_open_s"):
            video_writer = VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height,
                                       video_metadata.video_fps_exact, encoder=encoder, encoder_options=encoder_options,
                                       time_base=video_metadata.time_base, mp4_fast_start=mp4_fast_start)
        with profiler.measure("restore_write_loop_s"):
            frame_restorer_iterator = iter(frame_restorer)
            while True:
                with profiler.measure("frame_restorer_next_s"):
                    try:
                        elem = next(frame_restorer_iterator)
                    except StopIteration:
                        break
                if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                    success = False
                    frame_restorer_progressbar.error = True
                    if isinstance(elem, ErrorMarker):
                        raise RuntimeError(
                            f"Frame restoration pipeline failed: {elem}\n{elem.stack_trace}"
                        )
                    raise RuntimeError("Frame restoration pipeline stopped prematurely.")
                (restored_frame, restored_frame_pts) = elem
                with profiler.measure("video_writer_write_s"):
                    video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                frames_written += 1
                frame_restorer_progressbar.update()
    except (Exception, KeyboardInterrupt) as e:
        success = False
        if isinstance(e, KeyboardInterrupt):
            raise e
        else:
            error_message = str(e)
            print("Error on export", e)
    finally:
        if video_writer is not None:
            try:
                with profiler.measure("video_writer_release_s"):
                    video_writer.release()
            except Exception as e:
                success = False
                error_message = str(e)
                print("Error on export", e)
        with profiler.measure("frame_restorer_stop_s"):
            frame_restorer.stop()
        frame_restorer_profile = frame_restorer.get_last_profile()
        frame_restorer_progressbar.close(ensure_completed_bar=success)

    if success:
        if _is_full_video_passthrough(frame_restorer_profile):
            print(_("No mosaic clips detected, preserving original video"))
            with profiler.measure("output_passthrough_total_s"):
                input_extension = os.path.splitext(input_path)[1].lower()
                output_extension = os.path.splitext(output_path)[1].lower()
                input_audio_codec = audio_utils.get_audio_codec(video_metadata.video_file)
                # If the final output keeps the original suffix, preserve the source bytes directly.
                # This is required for batch workflows that intentionally keep ".webm" file names even
                # when the original container/codec combination is non-standard for that extension.
                can_copy_original = input_extension == output_extension
                passthrough_reason = "matching_extension_direct_copy" if can_copy_original else None
                if not can_copy_original:
                    can_copy_original, passthrough_reason = audio_utils.can_copy_video_file_to_output(
                        video_metadata.codec_name,
                        input_audio_codec,
                        output_path,
                    )
                if can_copy_original:
                    audio_profile = audio_utils.copy_or_remux_video_file(input_path, output_path)
                    audio_profile["passthrough_original_video"] = True
                    audio_profile["passthrough_reason"] = passthrough_reason
                    _cleanup_intermediate_outputs(
                        video_tmp_file_output_path=video_tmp_file_output_path,
                        working_output_path=encoded_output_path,
                        output_path=output_path,
                    )
                else:
                    print(
                        _("Preserving original video skipped, falling back to encoded passthrough"),
                        passthrough_reason,
                    )
                    audio_profile = audio_utils.combine_audio_video_files(
                        video_metadata,
                        video_tmp_file_output_path,
                        encoded_output_path,
                    )
                    audio_profile["passthrough_original_video"] = False
                    audio_profile["passthrough_fallback"] = "encoded_passthrough"
                    audio_profile["passthrough_reason"] = passthrough_reason
        else:
            print(_("Processing audio"))
            with profiler.measure("audio_mux_total_s"):
                audio_profile = audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, encoded_output_path)
        if (
            encoded_output_path != output_path
            and os.path.exists(encoded_output_path)
        ):
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(encoded_output_path, output_path)
    else:
        _cleanup_intermediate_outputs(
            video_tmp_file_output_path=video_tmp_file_output_path,
            working_output_path=encoded_output_path,
            output_path=output_path,
        )

    report: dict[str, object] = {
        "input_path": input_path,
        "output_path": output_path,
        "success": success,
        "frames_written": frames_written,
        "video_metadata": _video_metadata_to_report(video_metadata),
        "stages": profiler.snapshot(total_s=perf_counter() - started_at),
        "frame_restorer": frame_restorer_profile,
        "audio_mux": audio_profile,
        "passthrough_original_video": bool(audio_profile.get("passthrough_original_video")),
        "output_exists": os.path.exists(output_path),
        "working_output_path": encoded_output_path if encoded_output_path != output_path else None,
        "error_message": error_message,
    }
    if os.path.exists(output_path):
        report["output_size_bytes"] = os.path.getsize(output_path)
    return report

def main():
    command_profiler = WallClockProfiler()
    command_profiler.add_duration("cli_bootstrap_import_s", perf_counter() - _CLI_PROCESS_START)

    with command_profiler.measure("cli_setup_argparser_s"):
        argparser = setup_argparser()
    with command_profiler.measure("cli_parse_args_s"):
        args = argparser.parse_args()

    if args.version:
        print("Lada: ", VERSION)
        sys.exit(0)
    if args.list_encoders:
        utils.dump_encoders()
        sys.exit(0)
    if args.list_mosaic_detection_models:
        utils.dump_available_detection_models()
        sys.exit(0)
    if args.list_mosaic_restoration_models:
        utils.dump_available_restoration_models()
        sys.exit(0)
    if args.list_devices:
        utils.dump_torch_devices()
        sys.exit(0)
    if args.list_encoding_presets:
        utils.dump_available_encoding_presets()
        sys.exit(0)
    if args.list_encoder_options:
        utils.dump_encoder_options(args.list_encoder_options)
        sys.exit(0)
    if args.help or not args.input:
        argparser.print_help()
        sys.exit(0)

    with command_profiler.measure("cli_compute_target_resolution_s"):
        args.device = normalize_compute_target_id(args.device)
        compute_target = get_compute_target(args.device, include_experimental=True)
    if compute_target is None:
        print(_("Unknown compute target '{target}'. Use \"--list-devices\" to inspect supported targets.").format(target=args.device))
        sys.exit(1)
    if compute_target_issue := describe_compute_target_issue(args.device):
        print(compute_target_issue)
        sys.exit(1)
    if args.fp16 is None:
        args.fp16 = default_fp16_enabled_for_compute_target(args.device)

    with command_profiler.measure("cli_argument_validation_s"):
        if "{orig_file_name}" not in args.output_file_pattern or "." not in args.output_file_pattern:
            print(_("Invalid file name pattern. It must include the template string '{orig_file_name}' and a file extension"))
            sys.exit(1)
        if os.path.isdir(args.input) and args.output is not None and os.path.isfile(args.output):
            print(_("Invalid output directory. If input is a directory then --output must also be set to a directory"))
            sys.exit(1)
        if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
            print(_("Invalid input. No file or directory at {input_path}").format(input_path=args.input))
            sys.exit(1)
        if args.temporary_directory and not os.path.isdir(args.temporary_directory):
            print(_("Temporary directory {temporary_path} doesn't exist. Creating...").format(temporary_path=args.temporary_directory))
            os.makedirs(args.temporary_directory)
        if args.working_output_extension and not args.working_output_extension.startswith("."):
            args.working_output_extension = "." + args.working_output_extension
        if args.retry_count < 0:
            print(_("Invalid retry count"))
            sys.exit(1)
        if args.retry_delay_seconds < 0:
            print(_("Invalid retry delay"))
            sys.exit(1)
        if args.max_files < 0:
            print(_("Invalid max files value"))
            sys.exit(1)
        if args.backup_root and os.path.isfile(args.backup_root):
            print(_("Invalid backup directory"))
            sys.exit(1)

    use_batch_runner = (
        os.path.isdir(args.input)
        and (
            args.recursive
            or args.preserve_relative_paths
            or args.backup_root is not None
            or args.batch_state_path is not None
            or args.force_backup
            or args.force_reconvert
            or args.retry_count > 0
            or args.retry_failed_first
            or args.working_output_extension is not None
            or args.dry_run
            or args.max_files > 0
        )
    )

    with command_profiler.measure("cli_model_path_resolution_s"):
        if detection_modelfile := ModelFiles.get_detection_model_by_name(args.mosaic_detection_model):
            mosaic_detection_model_path = detection_modelfile.path
        elif os.path.isfile(args.mosaic_detection_model):
            mosaic_detection_model_path = args.mosaic_detection_model
        else:
            print(_("Invalid mosaic detection model"))
            sys.exit(1)

        if restoration_modelfile := ModelFiles.get_restoration_model_by_name(args.mosaic_restoration_model):
            mosaic_restoration_model_name = args.mosaic_restoration_model
            mosaic_restoration_model_path = restoration_modelfile.path
        elif os.path.isfile(args.mosaic_restoration_model):
            mosaic_restoration_model_path = args.mosaic_restoration_model
            mosaic_restoration_model_name = 'basicvsrpp' # Assume custom model is basicvsrpp. DeepMosaics custom path is not supported
        else:
            print(_("Invalid mosaic restoration model"))
            sys.exit(1)

    with command_profiler.measure("cli_encoder_resolution_s"):
        encoder = None
        encoder_options = None
        if args.encoder:
            encoder = args.encoder
            encoder_options = args.encoder_options if args.encoder_options else ''
        elif args.encoding_preset:
            encoding_presets = video_utils.get_encoding_presets()
            found = False
            for preset in encoding_presets:
                if preset.name == args.encoding_preset:
                    found = True
                    encoder = preset.encoder_name
                    encoder_options = preset.encoder_options
                    break
            if not found:
                print(_("Invalid encoding preset"))
                sys.exit(1)
        else:
            print(_('Either "--encoding-preset" or "--encoder" together with "--encoder-options" must be used'))
            sys.exit(1)
    assert encoder is not None and encoder_options is not None

    with command_profiler.measure("cli_torch_device_resolution_s"):
        device = (
            resolve_torch_device(args.device)
            if compute_target.torch_device is not None
            else torch.device("cpu")
        )
    mosaic_detection_model = None
    mosaic_restoration_model = None
    preferred_pad_mode = None
    if not (use_batch_runner and args.dry_run):
        with command_profiler.measure("cli_model_load_s"):
            loaded_models = load_models(
                args.device, compute_target.torch_device and torch.device(compute_target.torch_device), mosaic_restoration_model_name, mosaic_restoration_model_path, args.mosaic_restoration_config_path,
                mosaic_detection_model_path, args.fp16, args.detect_face_mosaics
            )
            mosaic_detection_model = loaded_models.detection_model
            mosaic_restoration_model = loaded_models.restoration_model
            preferred_pad_mode = loaded_models.preferred_pad_mode

    with command_profiler.measure("cli_input_output_setup_s"):
        if use_batch_runner:
            input_files = []
            output_files = []
        else:
            input_files, output_files = utils.setup_input_and_output_paths(args.input, args.output, args.output_file_pattern)

    single_file_input = (len(input_files) == 1) if not use_batch_runner else False
    file_reports: list[dict[str, object]] = []

    batch_failed_count = 0
    with command_profiler.measure("cli_process_inputs_s"):
        if use_batch_runner:
            batch_output_root = args.output if args.output else args.input
            if args.dry_run:
                def batch_process_file(**kwargs):
                    raise RuntimeError(f"Dry-run unexpectedly attempted to process a file: {kwargs.get('input_path')}")
            else:
                def batch_process_file(**kwargs):
                    return process_video_file(
                        temp_dir_path=args.temporary_directory,
                        device=device,
                        mosaic_restoration_model=mosaic_restoration_model,
                        mosaic_detection_model=mosaic_detection_model,
                        mosaic_restoration_model_name=mosaic_restoration_model_name,
                        preferred_pad_mode=preferred_pad_mode,
                        max_clip_length=args.max_clip_length,
                        encoder=encoder,
                        encoder_options=encoder_options,
                        mp4_fast_start=args.mp4_fast_start,
                        **kwargs,
                    )
            batch_result = batch_runner.run_batch(
                input_root=args.input,
                output_root=batch_output_root,
                output_file_pattern=args.output_file_pattern,
                process_file=batch_process_file,
                recursive=args.recursive,
                preserve_relative_paths=args.preserve_relative_paths,
                backup_root=args.backup_root,
                state_path=args.batch_state_path,
                force_backup=args.force_backup,
                force_reconvert=args.force_reconvert,
                retry_count=args.retry_count,
                retry_delay_seconds=args.retry_delay_seconds,
                retry_failed_first=args.retry_failed_first,
                working_output_extension=args.working_output_extension,
                dry_run=args.dry_run,
                max_files=args.max_files,
            )
            file_reports = batch_result.file_reports
            batch_failed_count = batch_result.failed_count
        else:
            for input_path, output_path in zip(input_files, output_files):
                if not single_file_input:
                    print(f"{os.path.basename(input_path)}:")
                try:
                    file_reports.append(
                        process_video_file(
                            input_path=input_path,
                            output_path=output_path,
                            temp_dir_path=args.temporary_directory,
                            device=device,
                            mosaic_restoration_model=mosaic_restoration_model,
                            mosaic_detection_model=mosaic_detection_model,
                            mosaic_restoration_model_name=mosaic_restoration_model_name,
                            preferred_pad_mode=preferred_pad_mode,
                            max_clip_length=args.max_clip_length,
                            encoder=encoder,
                            encoder_options=encoder_options,
                            mp4_fast_start=args.mp4_fast_start,
                        )
                    )
                except KeyboardInterrupt:
                    print(_("Received Ctrl-C, stopping restoration."))
                    break

    command_total_s = perf_counter() - _CLI_PROCESS_START
    command_report = {
        "argv": list(sys.argv),
        "command_total_s": command_total_s,
        "timings": command_profiler.snapshot(total_s=command_total_s),
        "compute_target": _compute_target_to_report(compute_target),
        "device": str(device),
        "fp16": bool(args.fp16),
        "encoder": {
            "name": encoder,
            "options": encoder_options,
            "mp4_fast_start": args.mp4_fast_start,
        },
        "inputs": {
            "input_root": args.input,
            "output_root": args.output,
            "temporary_directory": args.temporary_directory,
            "single_file_input": single_file_input,
        },
        "files": file_reports,
    }

    if args.timing_report_path:
        _write_timing_report(args.timing_report_path, command_report)
        print(_("Timing report written to {path}").format(path=args.timing_report_path))

    if batch_failed_count > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
