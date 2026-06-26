import os, torch, cv2, torchvision, subprocess, librosa, tempfile, face_alignment
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
from transformers import Wav2Vec2FeatureExtractor

from models.float.FLOAT import FLOAT


@dataclass
class InferenceConfig:
    # checkpoint
    ckpt_path: str = "./checkpoints/float.pth"

    # pretrained model paths
    wav2vec_model_path: str = "./checkpoints/wav2vec2-base-960h"
    audio2emotion_path: str = "./checkpoints/wav2vec-english-speech-emotion-recognition"

    # video
    input_size: int = 512
    input_nc: int = 3
    fps: float = 25.0

    # audio
    sampling_rate: int = 16000
    audio_marcing: int = 2
    wav2vec_sec: float = 2.0
    attention_window: int = 2
    only_last_features: bool = False
    average_emotion: bool = False

    # dropout
    audio_dropout_prob: float = 0.1
    ref_dropout_prob: float = 0.1
    emotion_dropout_prob: float = 0.1

    # model dimensions
    style_dim: int = 512
    dim_a: int = 512
    dim_w: int = 512
    dim_h: int = 1024
    dim_m: int = 20
    dim_e: int = 7

    # FMT
    fmt_depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    no_learned_pe: bool = False
    num_prev_frames: int = 10
    max_grad_norm: float = 1.0

    # ODE
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5
    nfe: int = 10
    torchdiffeq_ode_method: str = "euler"

    # CFG defaults (can be overridden per run_inference call)
    a_cfg_scale: float = 2.0
    e_cfg_scale: float = 1.0
    r_cfg_scale: float = 1.0

    # diffusion (ablation)
    n_diff_steps: int = 500
    diff_schedule: str = "cosine"
    diffusion_mode: str = "sample"

    # seed
    seed: int = 15
    fix_noise_seed: bool = False

    # device
    rank: Union[int, str] = 0
    ngpus: int = 1


class DataProcessor:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.fps = config.fps
        self.sampling_rate = config.sampling_rate
        self.input_size = config.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.wav2vec_model_path, local_files_only=True
        )
        self.transform = A.Compose([
            A.Resize(height=config.input_size, width=config.input_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A_pytorch.ToTensorV2(),
        ])

    @torch.no_grad()
    def process_img(self, img: np.ndarray) -> np.ndarray:
        mult = 360. / img.shape[0]
        resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult,
                                 interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [(int(x1/mult), int(y1/mult), int(x2/mult), int(y2/mult), score)
                  for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0]

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * 1.6)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs

        crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
        crop_img = cv2.resize(crop_img, dsize=(self.input_size, self.input_size),
                              interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return crop_img

    def default_img_loader(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sampling_rate, return_tensors='pt'
        ).input_values[0]

    def preprocess(self, ref_path: str, audio_path: str, no_crop: bool) -> dict:
        s = self.default_img_loader(ref_path)
        if not no_crop:
            s = self.process_img(s)
        s = self.transform(image=s)['image'].unsqueeze(0)
        a = self.default_aud_loader(audio_path).unsqueeze(0)
        return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
    def __init__(self, config: InferenceConfig):
        torch.cuda.empty_cache()
        self.config = config

        self._load_model()
        self._load_weights()
        self.G.to(config.rank)
        self.G.eval()

        self.data_processor = DataProcessor(config)

    def _load_model(self) -> None:
        self.G = FLOAT(self.config)

    def _load_weights(self) -> None:
        state_dict = torch.load(self.config.ckpt_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for name, param in self.G.named_parameters():
                if name in state_dict:
                    param.copy_(state_dict[name].to(self.config.rank))
                elif "wav2vec2" in name:
                    pass
                else:
                    print(f"! Warning: {name} not found in checkpoint.")
        del state_dict

    def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
        import shutil as _shutil
        vid = vid_target_recon.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
        vid = ((vid + 1) / 2 * 255).byte()
        T, H, W, C = vid.shape
        raw_frames = vid.numpy().tobytes()
        tmp_raw = None
        tmp_muxed = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                tmp_raw = f.name
            r = subprocess.run(
                ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', f'{W}x{H}', '-r', str(self.config.fps),
                 '-i', 'pipe:0',
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                 '-y', tmp_raw],
                input=raw_frames, capture_output=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f'ffmpeg encode failed: {r.stderr.decode()[:400]}')
            if audio_path is not None:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                    tmp_muxed = f.name
                r = subprocess.run(
                    ['ffmpeg', '-i', tmp_raw, '-i', audio_path,
                     '-c:v', 'copy', '-c:a', 'aac',
                     '-shortest', '-y', tmp_muxed],
                    capture_output=True,
                )
                if r.returncode != 0:
                    _shutil.copyfile(tmp_raw, video_path)
                else:
                    _shutil.copyfile(tmp_muxed, video_path)
            else:
                _shutil.copyfile(tmp_raw, video_path)
        finally:
            if tmp_raw and os.path.exists(tmp_raw):
                os.unlink(tmp_raw)
            if tmp_muxed and os.path.exists(tmp_muxed):
                os.unlink(tmp_muxed)
        return video_path

    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str,
        ref_path: str,
        audio_path: str,
        a_cfg_scale: Optional[float] = None,
        r_cfg_scale: Optional[float] = None,
        e_cfg_scale: Optional[float] = None,
        emo: Optional[str] = None,
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25,
        verbose: bool = False,
    ) -> str:
        """Run inference for a single (image, audio) pair.

        cfg_scale parameters default to None, which falls back to the values
        set in InferenceConfig, so you can set them once at construction time
        and override per-call only when needed.
        """
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)
        if verbose:
            print("> [Done] Preprocess.")

        d_hat = self.G.inference(
            data=data,
            a_cfg_scale=a_cfg_scale,
            r_cfg_scale=r_cfg_scale,
            e_cfg_scale=e_cfg_scale,
            emo=emo,
            nfe=nfe,
            seed=seed,
        )['d_hat']

        res_video_path = self.save_video(d_hat, res_video_path, audio_path)
        if verbose:
            print(f"> [Done] result saved at {res_video_path}")
        return res_video_path
