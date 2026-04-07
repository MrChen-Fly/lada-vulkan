# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from lada.cli import utils


@dataclass(frozen=True)
class BatchTask:
    input_path: str
    relative_path: str
    output_path: str
    working_output_path: str | None
    backup_path: str | None
    is_retry_priority: bool = False


@dataclass(frozen=True)
class BatchRunResult:
    file_reports: list[dict[str, object]]
    converted_count: int
    skipped_count: int
    failed_count: int
    state_path: str | None


def _get_now_iso_timestamp() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat()


def _load_json_file(path: str) -> dict[str, object] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None


def _save_json_file(path: str, payload: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    rounded_seconds = int(round(seconds))
    minutes, seconds = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _test_backup_file_ready(source_path: str, backup_path: str) -> bool:
    if not os.path.exists(backup_path):
        return False
    source_stat = os.stat(source_path)
    backup_stat = os.stat(backup_path)
    return (
        source_stat.st_size == backup_stat.st_size
        and int(source_stat.st_mtime) == int(backup_stat.st_mtime)
    )


def _copy_backup_file(source_path: str, backup_path: str) -> None:
    Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)


def _count_existing_outputs(tasks: list[BatchTask]) -> int:
    return sum(1 for task in tasks if os.path.exists(task.output_path))


def _count_existing_backups(tasks: list[BatchTask]) -> int:
    count = 0
    for task in tasks:
        if task.backup_path and _test_backup_file_ready(task.input_path, task.backup_path):
            count += 1
    return count


def _new_batch_state(
    *,
    input_root: str,
    output_root: str,
    backup_root: str | None,
    state_path: str,
    total_files: int,
    existing_output_count: int,
    existing_backup_count: int,
    loaded_state: dict[str, Any] | None,
    dry_run: bool,
) -> dict[str, object]:
    history_attempt_count = 0
    history_attempt_seconds = 0.0
    history_failed_files: list[str] = []

    if loaded_state is not None:
        history = loaded_state.get('history') or {}
        if isinstance(history.get('conversionAttempts'), int):
            history_attempt_count = history['conversionAttempts']
        if isinstance(history.get('conversionSeconds'), (int, float)):
            history_attempt_seconds = float(history['conversionSeconds'])
        if isinstance(history.get('failedFiles'), list):
            history_failed_files = [
                str(item) for item in history['failedFiles'] if isinstance(item, str) and item.strip()
            ]

    return {
        'version': 1,
        'statePath': state_path,
        'inputRoot': input_root,
        'outputRoot': output_root,
        'backupRoot': backup_root,
        'totalFiles': total_files,
        'createdAt': _get_now_iso_timestamp(),
        'updatedAt': _get_now_iso_timestamp(),
        'history': {
            'conversionAttempts': history_attempt_count,
            'conversionSeconds': history_attempt_seconds,
            'failedFiles': list(dict.fromkeys(history_failed_files)),
        },
        'run': {
            'startedAt': _get_now_iso_timestamp(),
            'phase': 'initializing',
            'dryRun': dry_run,
            'currentFile': None,
            'currentIndex': 0,
            'outputReadyAtStart': existing_output_count,
            'backupReadyAtStart': existing_backup_count,
            'converted': 0,
            'skipped': 0,
            'failed': 0,
            'backupCopied': 0,
            'backupSkipped': 0,
            'lastCompletedFile': None,
            'lastFailure': None,
        },
    }


def _remove_failed_relative_path(state: dict[str, object], relative_path: str) -> None:
    history = state['history']
    history['failedFiles'] = [
        path for path in history.get('failedFiles', []) if path.lower() != relative_path.lower()
    ]


def _add_failed_relative_path(state: dict[str, object], relative_path: str) -> None:
    history = state['history']
    failed_files = list(history.get('failedFiles', []))
    if relative_path.lower() in {path.lower() for path in failed_files}:
        return
    failed_files.append(relative_path)
    history['failedFiles'] = failed_files


def _get_average_attempt_seconds(state: dict[str, object]) -> float | None:
    history = state['history']
    attempts = int(history.get('conversionAttempts', 0))
    if attempts <= 0:
        return None
    return float(history.get('conversionSeconds', 0.0)) / attempts


def _get_ready_output_count(state: dict[str, object]) -> int:
    run = state['run']
    return int(run.get('outputReadyAtStart', 0)) + int(run.get('converted', 0))


def _get_remaining_output_count(state: dict[str, object]) -> int:
    return max(int(state.get('totalFiles', 0)) - _get_ready_output_count(state), 0)


def _get_eta_text(state: dict[str, object]) -> str:
    average = _get_average_attempt_seconds(state)
    if average is None:
        return '--:--'
    return _format_duration(average * _get_remaining_output_count(state))


def _build_tasks(
    *,
    input_root: str,
    output_root: str,
    output_file_pattern: str,
    backup_root: str | None,
    recursive: bool,
    preserve_relative_paths: bool,
    working_output_extension: str | None,
    max_files: int,
) -> list[BatchTask]:
    input_files = utils.list_video_files(input_root, recursive=recursive)
    if max_files > 0:
        input_files = input_files[:max_files]
    tasks: list[BatchTask] = []
    for input_path in input_files:
        relative_path = os.path.relpath(input_path, input_root)
        output_path = utils.build_output_file_path(
            input_file_path=input_path,
            output_directory=output_root,
            output_file_pattern=output_file_pattern,
            input_root_directory=input_root,
            preserve_relative_paths=preserve_relative_paths,
        )
        working_output_path = None
        if working_output_extension:
            base, _ = os.path.splitext(output_path)
            working_output_path = base + working_output_extension
            if os.path.normcase(working_output_path) == os.path.normcase(output_path):
                working_output_path = None
        backup_path = os.path.join(backup_root, relative_path) if backup_root else None
        tasks.append(
            BatchTask(
                input_path=input_path,
                relative_path=relative_path,
                output_path=output_path,
                working_output_path=working_output_path,
                backup_path=backup_path,
            )
        )
    return tasks


def _get_retry_priority_relative_paths(
    loaded_state: dict[str, Any] | None,
    tasks: list[BatchTask],
) -> list[str]:
    persisted_failed = []
    if loaded_state is not None:
        history = loaded_state.get('history') or {}
        if isinstance(history.get('failedFiles'), list):
            persisted_failed = [
                str(item) for item in history['failedFiles'] if isinstance(item, str) and item.strip()
            ]
    if persisted_failed:
        existing_relatives = {task.relative_path.lower(): task.relative_path for task in tasks}
        resolved: list[str] = []
        for relative_path in persisted_failed:
            original = existing_relatives.get(relative_path.lower())
            if original is not None:
                resolved.append(original)
        return list(dict.fromkeys(resolved))

    if loaded_state is None:
        return []
    run = loaded_state.get('run') or {}
    if bool(run.get('dryRun')):
        return []
    current_index = int(run.get('currentIndex', 0) or 0)
    if current_index <= 0:
        return []
    retry_relatives: list[str] = []
    for task in tasks[: min(current_index, len(tasks))]:
        if not os.path.exists(task.output_path):
            retry_relatives.append(task.relative_path)
    return list(dict.fromkeys(retry_relatives))


def _order_tasks_failed_first(tasks: list[BatchTask], retry_relative_paths: list[str]) -> list[BatchTask]:
    retry_lookup = {path.lower() for path in retry_relative_paths}
    priority: list[BatchTask] = []
    regular: list[BatchTask] = []
    for task in tasks:
        if task.relative_path.lower() in retry_lookup:
            priority.append(BatchTask(**{**task.__dict__, 'is_retry_priority': True}))
        else:
            regular.append(task)
    return priority + regular


class BatchRunnerError(RuntimeError):
    pass


def run_batch(
    *,
    input_root: str,
    output_root: str,
    output_file_pattern: str,
    process_file: Callable[..., dict[str, object]],
    recursive: bool,
    preserve_relative_paths: bool,
    backup_root: str | None,
    state_path: str | None,
    force_backup: bool,
    force_reconvert: bool,
    retry_count: int,
    retry_delay_seconds: int,
    retry_failed_first: bool,
    working_output_extension: str | None,
    dry_run: bool,
    max_files: int = 0,
) -> BatchRunResult:
    tasks = _build_tasks(
        input_root=input_root,
        output_root=output_root,
        output_file_pattern=output_file_pattern,
        backup_root=backup_root,
        recursive=recursive,
        preserve_relative_paths=preserve_relative_paths,
        working_output_extension=working_output_extension,
        max_files=max_files,
    )
    if not tasks:
        raise BatchRunnerError(f'No video files found under {input_root}')

    loaded_state = _load_json_file(state_path) if state_path else None
    retry_priority_relative_paths = _get_retry_priority_relative_paths(loaded_state, tasks) if retry_failed_first else []
    ordered_tasks = _order_tasks_failed_first(tasks, retry_priority_relative_paths)

    existing_output_count = 0 if force_reconvert else _count_existing_outputs(tasks)
    existing_backup_count = 0 if force_backup or not backup_root else _count_existing_backups(tasks)
    state = _new_batch_state(
        input_root=input_root,
        output_root=output_root,
        backup_root=backup_root,
        state_path=state_path or '',
        total_files=len(tasks),
        existing_output_count=existing_output_count,
        existing_backup_count=existing_backup_count,
        loaded_state=loaded_state,
        dry_run=dry_run,
    )
    state['history']['failedFiles'] = retry_priority_relative_paths

    if state_path:
        _save_json_file(state_path, state)

    print(f'Source root : {input_root}')
    print(f'Output root : {output_root}')
    if backup_root:
        print(f'Backup root : {backup_root}')
    if state_path:
        print(f'State file  : {state_path}')
    print(f'Files found : {len(tasks)}')
    print(
        f'Resume      : backup ready {existing_backup_count}/{len(tasks)}, '
        f'output ready {existing_output_count}/{len(tasks)}'
    )
    if retry_failed_first:
        print(f'Retry queue : {len(retry_priority_relative_paths)} failed file(s) will run before new files')
    print(f'ETA         : {_get_eta_text(state)}')
    if dry_run:
        print('Mode        : DryRun')

    backup_copied = 0
    backup_skipped = 0
    if backup_root:
        state['run']['phase'] = 'backup'
        if state_path:
            _save_json_file(state_path, state)
        for index, task in enumerate(tasks, start=1):
            state['run']['currentFile'] = task.relative_path
            state['run']['currentIndex'] = index
            if task.backup_path is None:
                continue
            if not force_backup and _test_backup_file_ready(task.input_path, task.backup_path):
                backup_skipped += 1
                state['run']['backupSkipped'] = backup_skipped
                if state_path and (index % 25 == 0 or index == len(tasks)):
                    _save_json_file(state_path, state)
                continue
            if not dry_run:
                _copy_backup_file(task.input_path, task.backup_path)
            backup_copied += 1
            state['run']['backupCopied'] = backup_copied
            if state_path and (index % 25 == 0 or index == len(tasks)):
                _save_json_file(state_path, state)

    file_reports: list[dict[str, object]] = []
    converted_count = 0
    skipped_count = 0
    failed_count = 0
    state['run']['phase'] = 'convert'
    if state_path:
        _save_json_file(state_path, state)

    for index, task in enumerate(ordered_tasks, start=1):
        state['run']['currentFile'] = task.relative_path
        state['run']['currentIndex'] = index

        if not force_reconvert and os.path.exists(task.output_path):
            skipped_count += 1
            state['run']['skipped'] = skipped_count
            _remove_failed_relative_path(state, task.relative_path)
            if state_path and (index % 25 == 0 or index == len(ordered_tasks)):
                _save_json_file(state_path, state)
            continue

        Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            print('[DryRun] Would convert:')
            print(f'  input : {task.input_path}')
            if task.working_output_path:
                print(f'  work  : {task.working_output_path}')
            print(f'  final : {task.output_path}')
            converted_count += 1
            state['run']['converted'] = converted_count
            continue

        print(f'[{index}/{len(ordered_tasks)}] {task.relative_path}')
        failure_messages: list[str] = []
        attempt_limit = retry_count + 1
        conversion_succeeded = False
        report: dict[str, object] | None = None

        for attempt_number in range(1, attempt_limit + 1):
            attempt_started = perf_counter()
            try:
                if task.working_output_path and os.path.exists(task.working_output_path):
                    os.remove(task.working_output_path)
                if os.path.exists(task.output_path):
                    os.remove(task.output_path)
                report = process_file(
                    input_path=task.input_path,
                    output_path=task.output_path,
                    working_output_path=task.working_output_path,
                )
                attempt_seconds = perf_counter() - attempt_started
                state['history']['conversionAttempts'] += 1
                state['history']['conversionSeconds'] += attempt_seconds
                file_reports.append(report)
                if bool(report.get('success')) and bool(report.get('output_exists')) and os.path.exists(task.output_path):
                    converted_count += 1
                    state['run']['converted'] = converted_count
                    state['run']['lastCompletedFile'] = task.relative_path
                    state['run']['lastFailure'] = None
                    _remove_failed_relative_path(state, task.relative_path)
                    conversion_succeeded = True
                    break
                failure_message = str(
                    report.get('error_message') or f'Lada completed but no output was written: {task.output_path}'
                )
                failure_messages.append(f'attempt {attempt_number}/{attempt_limit}: {failure_message}')
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                attempt_seconds = perf_counter() - attempt_started
                state['history']['conversionAttempts'] += 1
                state['history']['conversionSeconds'] += attempt_seconds
                failure_messages.append(f'attempt {attempt_number}/{attempt_limit}: {exc}')
            if task.working_output_path and os.path.exists(task.working_output_path):
                os.remove(task.working_output_path)
            if attempt_number < attempt_limit:
                print(f"Retrying '{task.relative_path}' ({attempt_number}/{retry_count}) after error: {failure_messages[-1]}")
                if state_path:
                    _save_json_file(state_path, state)
                if retry_delay_seconds > 0:
                    import time

                    time.sleep(retry_delay_seconds)
                continue

            failed_count += 1
            state['run']['failed'] = failed_count
            _add_failed_relative_path(state, task.relative_path)
            state['run']['lastFailure'] = {
                'file': task.relative_path,
                'message': ' | '.join(failure_messages),
                'at': _get_now_iso_timestamp(),
            }
            print(f"WARNING: Conversion failed for '{task.input_path}': {' | '.join(failure_messages)}")

        if state_path:
            _save_json_file(state_path, state)
        if not conversion_succeeded:
            continue

    if dry_run:
        state['run']['phase'] = 'completed_dry_run'
    else:
        state['run']['phase'] = 'completed_with_failures' if failed_count > 0 else 'completed'
    state['run']['currentFile'] = None
    if state_path:
        _save_json_file(state_path, state)

    print('')
    if backup_root:
        backup_label = 'planned' if dry_run else 'copied '
        print('Backup summary')
        print(f'  {backup_label}: {backup_copied}')
        print(f'  skipped: {backup_skipped}')
        print('')
    conversion_label = 'planned  ' if dry_run else 'converted'
    print('Conversion summary')
    print(f'  {conversion_label}: {converted_count}')
    print(f'  skipped  : {skipped_count}')
    print(f'  failed   : {failed_count}')
    print(f'  ready    : {_get_ready_output_count(state)}/{len(tasks)}')
    print(f'  remaining: {_get_remaining_output_count(state)}')
    average = _get_average_attempt_seconds(state)
    print(f"  avg/file : {_format_duration(average) if average is not None else '--:--'}")

    return BatchRunResult(
        file_reports=file_reports,
        converted_count=converted_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        state_path=state_path,
    )
