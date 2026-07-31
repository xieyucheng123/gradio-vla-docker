import argparse
import base64
import io
import time
import logging

import gradio as gr
import requests
import numpy as np
from PIL import Image

from config import (
    OPENVLA_API, ROBOTWIN_API,
    DEFAULT_MAX_STEPS, DEFAULT_TASK, DEFAULT_INSTRUCTION,
    API_TIMEOUT, LOOP_TIMEOUT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _img_from_b64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _img_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _frames_to_video(frames: list[bytes], fps: int = 5) -> str:
    import tempfile, os, subprocess
    if not frames:
        return ""
    tmpdir = tempfile.mkdtemp()
    for i, f in enumerate(frames):
        with open(os.path.join(tmpdir, f"frame_{i:04d}.png"), "wb") as fp:
            fp.write(f)
    out_path = os.path.join(tmpdir, "simulation.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(tmpdir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


# ── Tab 1: Single-step inference ─────────────────────────────────────────────

def single_step_infer(image, instruction, unnorm_key):
    if image is None:
        return "Please provide an image first.", ""
    if not instruction:
        return "Please provide an instruction.", ""
    try:
        img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        img_np = _img_to_np(img_pil.convert("RGB"))
        payload = {
            "image": img_np.tolist(),
            "instruction": instruction,
            "unnorm_key": unnorm_key or "libero_object",
        }
        t0 = time.time()
        resp = requests.post(f"{OPENVLA_API}/act", json=payload, timeout=API_TIMEOUT)
        dt = time.time() - t0
        if resp.status_code == 200:
            action = resp.json().get("action", [])
            text = f"Action (7-DoF): {action}\nInference time: {dt:.2f}s"
            return text, str(action)
        return f"Error {resp.status_code}: {resp.text}", ""
    except Exception as e:
        return f"Error: {e}", ""


def fetch_sim_image():
    try:
        resp = requests.get(f"{ROBOTWIN_API}/render", timeout=API_TIMEOUT)
        if resp.status_code == 200:
            return np.array(Image.open(io.BytesIO(resp.content)))
        return None
    except Exception as e:
        logger.error(f"fetch_sim_image: {e}")
        return None


# ── Tab 2: Closed-loop control ───────────────────────────────────────────────

def run_closed_loop(task, instruction, max_steps, seed, progress=gr.Progress()):
    if not task:
        return None, "Please select a task.", ""
    if not instruction:
        return None, "Please provide an instruction.", ""

    log_lines = []
    frames = []

    def log(msg):
        log_lines.append(msg)
        logger.info(msg)

    try:
        # 1. Reset simulation
        log(f"[Reset] task={task}, seed={seed}")
        resp = requests.post(
            f"{ROBOTWIN_API}/reset",
            json={"task": task, "seed": int(seed)},
            timeout=API_TIMEOUT,
        )
        if resp.status_code != 200:
            return None, f"Reset failed: {resp.text}", "\n".join(log_lines)
        log("[Reset] OK")

        # 2. Closed-loop
        success = False
        for step in range(int(max_steps)):
            progress((step + 1) / int(max_steps), desc=f"Step {step + 1}/{max_steps}")

            # 2a. Get observation + render
            obs_resp = requests.get(f"{ROBOTWIN_API}/obs", timeout=API_TIMEOUT)
            if obs_resp.status_code != 200:
                log(f"Step {step+1}: obs failed {obs_resp.status_code}")
                break
            obs_data = obs_resp.json()

            # Get rendered frame
            render_resp = requests.get(f"{ROBOTWIN_API}/render", timeout=API_TIMEOUT)
            if render_resp.status_code == 200:
                frames.append(render_resp.content)

            # 2b. Model inference
            img_b64 = obs_data.get("image", "")
            t0 = time.time()
            act_resp = requests.post(
                f"{OPENVLA_API}/act",
                json={
                    "image": img_b64,
                    "instruction": instruction,
                    "unnorm_key": "libero_object",
                },
                timeout=API_TIMEOUT,
            )
            dt = time.time() - t0
            if act_resp.status_code != 200:
                log(f"Step {step+1}: inference failed {act_resp.status_code}")
                break
            action = act_resp.json().get("action", [])
            log(f"Step {step+1}: action={[round(a, 3) for a in action]} ({dt:.2f}s)")

            # 2c. Execute action
            step_resp = requests.post(
                f"{ROBOTWIN_API}/step",
                json={"action": action},
                timeout=API_TIMEOUT,
            )
            if step_resp.status_code != 200:
                log(f"Step {step+1}: step failed {step_resp.status_code}")
                break
            result = step_resp.json()

            # 2d. Check termination
            if result.get("done", False):
                success = result.get("success", False)
                log(f"Step {step+1}: {'SUCCESS' if success else 'FAILED'} (done)")
                break

        # 3. Make video
        video_path = _frames_to_video(frames, fps=5) if frames else ""

        status = f"{'✅ Success' if success else '❌ Failed'} ({step+1} steps, {len(frames)} frames)"
        return video_path if video_path else None, status, "\n".join(log_lines)

    except Exception as e:
        logger.error(f"Closed-loop error: {e}", exc_info=True)
        return None, f"Error: {e}", "\n".join(log_lines)


# ── Tab 3: Batch evaluation ──────────────────────────────────────────────────

def run_evaluation(tasks, num_seeds, max_steps, progress=gr.Progress()):
    if not tasks:
        return "Please select at least one task.", ""

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    total = len(task_list) * int(num_seeds)
    done = 0
    results = {}
    all_logs = []

    for task in task_list:
        successes = 0
        for seed in range(int(num_seeds)):
            done += 1
            progress(done / total, desc=f"{task} seed {seed+1}/{num_seeds}")
            try:
                requests.post(f"{ROBOTWIN_API}/reset",
                              json={"task": task, "seed": seed}, timeout=API_TIMEOUT)
                success = False
                for step in range(int(max_steps)):
                    obs_resp = requests.get(f"{ROBOTWIN_API}/obs", timeout=API_TIMEOUT)
                    obs_data = obs_resp.json()
                    act_resp = requests.post(f"{OPENVLA_API}/act",
                        json={"image": obs_data.get("image", ""),
                              "instruction": DEFAULT_INSTRUCTION,
                              "unnorm_key": "libero_object"}, timeout=API_TIMEOUT)
                    action = act_resp.json().get("action", [])
                    step_resp = requests.post(f"{ROBOTWIN_API}/step",
                        json={"action": action}, timeout=API_TIMEOUT)
                    result = step_resp.json()
                    if result.get("done", False):
                        success = result.get("success", False)
                        break
                if success:
                    successes += 1
                all_logs.append(f"{task} seed={seed}: {'✅' if success else '❌'} ({step+1} steps)")
            except Exception as e:
                all_logs.append(f"{task} seed={seed}: ERROR {e}")

        sr = successes / int(num_seeds) * 100
        results[task] = (successes, int(num_seeds), sr)

    # Build summary
    lines = ["| Task | Success | Total | Rate |", "|------|---------|-------|------|"]
    total_sr = 0
    for task, (s, n, sr) in results.items():
        lines.append(f"| {task} | {s} | {n} | {sr:.1f}% |")
        total_sr += sr
    avg_sr = total_sr / len(results) if results else 0
    lines.append(f"| **Average** | - | - | **{avg_sr:.1f}%** |")

    return "\n".join(lines), "\n".join(all_logs)


# ── Health checks ────────────────────────────────────────────────────────────

def check_health():
    results = []
    for name, url in [("OpenVLA (0001)", OPENVLA_API), ("RoboTwin (0002)", ROBOTWIN_API)]:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            results.append(f"{name}: {'✅' if resp.status_code == 200 else '❌'}")
        except Exception:
            results.append(f"{name}: ❌ (unreachable)")
    return "\n".join(results)


def get_task_list():
    try:
        resp = requests.get(f"{ROBOTWIN_API}/tasks", timeout=API_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("tasks", [DEFAULT_TASK])
    except Exception:
        pass
    return [DEFAULT_TASK]


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="PickAgent VLA - Closed-Loop Control", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PickAgent VLA — 闭环控制演示平台")
        gr.Markdown("三层架构: Gradio (本机) → OpenVLA REST API (ECS 0001) + RoboTwin REST API (ECS 0002)")

        with gr.Row():
            health_btn = gr.Button("检查服务状态", size="sm")
            health_out = gr.Textbox(label="服务状态", interactive=False, lines=2)

        health_btn.click(fn=check_health, outputs=health_out)

        # ── Tab 1: Single-step ──
        with gr.Tab("单步推理"):
            gr.Markdown("### 单步推理（辅助理解）\n上传图片+指令 → 返回 7-DoF 动作向量")
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="观测图片", type="numpy")
                    instr_in = gr.Textbox(label="指令", value=DEFAULT_INSTRUCTION)
                    unnorm_in = gr.Textbox(label="Unnorm Key", value="libero_object")
                    with gr.Row():
                        infer_btn = gr.Button("推理", variant="primary")
                        fetch_btn = gr.Button("从仿真获取")
                with gr.Column():
                    out_text = gr.Textbox(label="推理结果", interactive=False, lines=5)
                    out_raw = gr.Textbox(label="原始动作", interactive=False, lines=2)

            infer_btn.click(fn=single_step_infer,
                            inputs=[img_in, instr_in, unnorm_in],
                            outputs=[out_text, out_raw])
            fetch_btn.click(fn=fetch_sim_image, outputs=img_in)

        # ── Tab 2: Closed-loop ──
        with gr.Tab("闭环控制"):
            gr.Markdown("### 闭环控制（核心功能）\n观测 → 推理 → 执行 → 循环，自动运行完整 episode")
            with gr.Row():
                with gr.Column(scale=1):
                    task_dd = gr.Dropdown(label="任务", choices=[DEFAULT_TASK], value=DEFAULT_TASK)
                    instr_loop = gr.Textbox(label="指令", value=DEFAULT_INSTRUCTION)
                    max_steps_in = gr.Slider(label="最大步数", minimum=10, maximum=300, value=DEFAULT_MAX_STEPS, step=10)
                    seed_in = gr.Number(label="随机种子", value=0, precision=0)
                    run_btn = gr.Button("▶ 运行闭环", variant="primary")
                with gr.Column(scale=2):
                    video_out = gr.Video(label="仿真视频")
                    status_out = gr.Textbox(label="结果", interactive=False)
                    log_out = gr.Textbox(label="运行日志", interactive=False, lines=15)

            run_btn.click(fn=run_closed_loop,
                          inputs=[task_dd, instr_loop, max_steps_in, seed_in],
                          outputs=[video_out, status_out, log_out])

        # ── Tab 3: Evaluation ──
        with gr.Tab("批量测评"):
            gr.Markdown("### 批量测评\n多任务 × 多种子 → 成功率统计")
            with gr.Row():
                with gr.Column(scale=1):
                    tasks_in = gr.Textbox(label="任务列表 (逗号分隔)", value=DEFAULT_TASK)
                    seeds_in = gr.Slider(label="种子数", minimum=1, maximum=50, value=5, step=1)
                    eval_steps_in = gr.Slider(label="最大步数", minimum=10, maximum=300, value=DEFAULT_MAX_STEPS, step=10)
                    eval_btn = gr.Button("▶ 开始测评", variant="primary")
                with gr.Column(scale=2):
                    eval_result = gr.Markdown(label="测评结果")
                    eval_log = gr.Textbox(label="详细日志", interactive=False, lines=15)

            eval_btn.click(fn=run_evaluation,
                          inputs=[tasks_in, seeds_in, eval_steps_in],
                          outputs=[eval_result, eval_log])

        # Load task list on startup
        demo.load(fn=lambda: gr.update(choices=get_task_list()), outputs=task_dd)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=True, prevent_thread_lock=False)
