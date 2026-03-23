"""
Gradio Web UI — 브라우저에서 ref image + pose video 업로드해서 바로 결과 확인
실행: python run_gradio.py
      → http://localhost:7860 에서 열기

참고: 최소 16GB VRAM 필요. VRAM 부족하면 해상도/프레임 줄여라.
"""

import sys
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

# ─── 레포 루트를 path에 추가 ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from tools.vid2pose import extract_pose_from_video
from scripts.pose2vid import load_models, build_pipeline, run_inference


CONFIG_PATH = "./configs/prompts/animation.yaml"


class AnimateController:
    def __init__(self):
        self.pipe = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16

    def load(self):
        """최초 1회만 모델 로드"""
        if self.pipe is not None:
            return
        print("Loading models... (최초 1회)")
        self.config = OmegaConf.load(CONFIG_PATH)
        vae, ref_unet, denoise_unet, pose_guider, image_enc, infer_config = load_models(
            self.config, self.weight_dtype, self.device
        )
        self.pipe = build_pipeline(
            vae, image_enc, ref_unet, denoise_unet, pose_guider,
            infer_config, self.weight_dtype, self.device
        )
        print("Models ready.")

    def animate(
        self,
        ref_image_path: str,
        driving_video_path: str,
        width: int,
        height: int,
        n_frames: int,
        steps: int,
        cfg: float,
        seed: int,
    ):
        """
        Gradio에서 호출되는 메인 함수.
        ref_image_path: 업로드된 이미지 경로
        driving_video_path: 업로드된 드라이빙 비디오 경로
        """
        if ref_image_path is None or driving_video_path is None:
            return None, "입력 파일을 모두 업로드해주세요."

        self.load()

        # ─── 포즈 추출 ────────────────────────────────────────────────────────
        pose_video_path = str(
            Path(driving_video_path).parent
            / (Path(driving_video_path).stem + "_kps.mp4")
        )
        print(f"Extracting pose from: {driving_video_path}")
        extract_pose_from_video(
            video_path=driving_video_path,
            output_path=pose_video_path,
            device=self.device,
        )

        # ─── inference ────────────────────────────────────────────────────────
        save_dir = Path("output/gradio") / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.manual_seed(seed)

        out_path = run_inference(
            pipe=self.pipe,
            ref_image_path=ref_image_path,
            pose_video_path=pose_video_path,
            width=width,
            height=height,
            n_frames=n_frames,
            steps=steps,
            cfg=cfg,
            generator=generator,
            save_dir=save_dir,
        )
        return out_path, f"완료: {out_path}"


controller = AnimateController()


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Moore-AnimateAnyone") as demo:
    gr.Markdown("## Moore-AnimateAnyone — Character Animation")
    gr.Markdown(
        "Reference Image + Driving Video를 업로드하면 해당 사람이 같은 동작으로 움직이는 영상을 생성합니다."
    )

    with gr.Row():
        with gr.Column():
            ref_image  = gr.Image(type="filepath", label="Reference Image (사람 이미지)")
            drive_video = gr.Video(label="Driving Video (동작 비디오)")

            with gr.Accordion("고급 설정", open=False):
                width    = gr.Slider(384, 768, value=512,  step=64,  label="Width")
                height   = gr.Slider(512, 1024, value=784, step=64,  label="Height")
                n_frames = gr.Slider(16,  128,  value=32,  step=8,   label="Frames")
                steps    = gr.Slider(10,  50,   value=30,  step=5,   label="DDIM Steps")
                cfg      = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="CFG Scale")
                seed     = gr.Number(value=42, label="Seed", precision=0)

            run_btn = gr.Button("Generate Animation", variant="primary")

        with gr.Column():
            output_video  = gr.Video(label="결과 (ref | pose | generated)")
            status_text   = gr.Textbox(label="상태", interactive=False)

    run_btn.click(
        fn=controller.animate,
        inputs=[ref_image, drive_video, width, height, n_frames, steps, cfg, seed],
        outputs=[output_video, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
