#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping

import os
import numpy as np


class WandbWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):
        self.wandb_run = None
        self.wandb_enabled = os.getenv("WANDB_ENABLE", "0") == "1"
        self.wandb_strict = os.getenv("WANDB_STRICT", "0") == "1"

        if self.wandb_enabled:
            try:
                import wandb

                # Prefer API key from env; otherwise use existing login state
                # and allow interactive login in TTY sessions.
                api_key = os.getenv("WANDB_API_KEY", "").strip()
                if api_key:
                    wandb.login(key=api_key, relogin=True)
                else:
                    logged_in = wandb.login(anonymous="never")
                    if not logged_in:
                        raise RuntimeError(
                            "W&B login required. Set WANDB_API_KEY or run `wandb login`."
                        )

                self.wandb_run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "ss-lite"),
                    entity=os.getenv("WANDB_ENTITY"),
                    name=os.getenv("WANDB_RUN_NAME"),
                    group=os.getenv("WANDB_RUN_GROUP"),
                    job_type=os.getenv("WANDB_JOB_TYPE"),
                    dir=log_dir if log_dir else None,
                    reinit=True,
                )
            except Exception as e:
                if self.wandb_strict:
                    raise RuntimeError(f"W&B init failed in strict mode: {e}") from e
                print(f"[WARN] W&B disabled: {e}")

    def __getattr__(self, item):
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int = None) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.log({tag: scalar_value}, step=global_step)
            except Exception:
                pass

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: Mapping[str, Any], global_step: int = None
    ) -> None:
        if self.wandb_run is None:
            return
        try:
            payload = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            self.wandb_run.log(payload, step=global_step)
        except Exception:
            pass

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int = None,
        dataformats: str = "CHW",
    ) -> None:
        if self.wandb_run is None:
            return
        try:
            import wandb

            img = np.asarray(img_tensor)
            fmt = dataformats.upper()
            if img.ndim == 3 and fmt == "CHW":
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            self.wandb_run.log({tag: wandb.Image(img)}, step=global_step)
        except Exception:
            pass

    def add_histogram(self, tag: str, values: Any, global_step: int = None) -> None:
        if self.wandb_run is None:
            return
        try:
            import wandb

            arr = np.asarray(values).reshape(-1)
            self.wandb_run.log({tag: wandb.Histogram(arr)}, step=global_step)
        except Exception:
            pass

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        if self.wandb_run is not None:
            try:
                import wandb

                video_np = np.asarray(images)
                # wandb.Video expects [T, C, H, W] for numpy inputs.
                if video_np.ndim == 4 and video_np.shape[-1] in (1, 3, 4):
                    video_np = np.transpose(video_np, (0, 3, 1, 2))
                if video_np.dtype != np.uint8:
                    if video_np.max() <= 1.0:
                        video_np = (video_np * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        video_np = video_np.clip(0, 255).astype(np.uint8)
                self.wandb_run.log(
                    {video_name: wandb.Video(video_np, fps=fps, format="mp4")},
                    step=step_idx,
                )
            except Exception as e:
                print(f"[WARN] W&B video log failed for '{video_name}' at step {step_idx}: {e}")

    def add_video_from_file(
        self, video_name: str, step_idx: int, video_path: str, fps: int = 10
    ) -> None:
        if self.wandb_run is not None:
            try:
                import wandb

                self.wandb_run.log(
                    {video_name: wandb.Video(video_path, fps=fps, format="mp4")},
                    step=step_idx,
                )
            except Exception as e:
                print(f"[WARN] W&B file video log failed for '{video_name}' at step {step_idx}: {e}")

    def flush(self) -> None:
        return

    def close(self) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
