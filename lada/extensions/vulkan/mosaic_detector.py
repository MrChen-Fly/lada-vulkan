from __future__ import annotations

from lada.restorationpipeline.mosaic_detector import Clip, MosaicDetector, Scene
from lada.utils.threading_utils import STOP_MARKER, StopMarker

from .clip_size_policy import resolve_clip_size_for_edge


class VulkanMosaicDetector(MosaicDetector):
    """Vulkan detector that promotes large scenes to larger restoration clips."""

    _SCENE_BORDER_RATIO = 0.06
    _SCENE_MIN_BORDER = 20
    _SAFE_CLIP_SIZE = 320

    def _estimate_scene_crop_max_edge(self, scene: Scene) -> int:
        max_crop_edge = 0
        for t, l, b, r in scene.boxes:
            height = int(b - t + 1)
            width = int(r - l + 1)
            edge = max(height, width)
            border = max(self._SCENE_MIN_BORDER, int(edge * self._SCENE_BORDER_RATIO))
            max_crop_edge = max(max_crop_edge, edge + border * 2)
        return max_crop_edge

    def _resolve_scene_clip_size(self, scene: Scene) -> int:
        clip_size = resolve_clip_size_for_edge(
            self._estimate_scene_crop_max_edge(scene),
            self.clip_size,
        )
        if not isinstance(self.clip_size, tuple):
            return clip_size

        sorted_clip_sizes = tuple(sorted(int(size) for size in self.clip_size))
        scene_length = max(len(scene), 1)
        pixel_budget = max(int(self.max_clip_length), 1) * self._SAFE_CLIP_SIZE * self._SAFE_CLIP_SIZE
        while (
            scene_length * clip_size * clip_size > pixel_budget
            and clip_size > sorted_clip_sizes[0]
        ):
            clip_index = sorted_clip_sizes.index(clip_size)
            clip_size = sorted_clip_sizes[clip_index - 1]
        return clip_size

    def _build_clip(self, scene: Scene, clip_id: int) -> Clip:
        return Clip(scene, self._resolve_scene_clip_size(scene), self.pad_mode, clip_id)

    def _create_clips_for_completed_scenes(self, scenes, frame_num, eof) -> StopMarker | None:
        completed_scenes = []
        for current_scene in scenes:
            if (current_scene.frame_end < frame_num or len(current_scene) >= self.max_clip_length or eof) and current_scene not in completed_scenes:
                completed_scenes.append(current_scene)
                other_scenes = [other for other in scenes if other != current_scene]
                for other_scene in other_scenes:
                    if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                        completed_scenes.append(other_scene)

        for completed_scene in sorted(completed_scenes, key=lambda s: s.frame_start):
            clip = self._build_clip(completed_scene, clip_id=self.clip_counter)
            self.mosaic_clip_queue.put(clip)
            if self.stop_requested:
                return STOP_MARKER
            scenes.remove(completed_scene)
            self.clip_counter += 1
        return None
