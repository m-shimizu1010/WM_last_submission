import torch
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
import numpy as np
import imageio

# 設定を読み込む
cfg = parse_cfg(Path('cfgs'))

# 環境を作成
env = make_env(cfg)

# デバッグ: cfgの状態を確認
print(f"DEBUG: cfg.task = {cfg.task}")
print(f"DEBUG: cfg.modality = {cfg.modality}")
print(f"DEBUG: cfg.obs_shape = {cfg.obs_shape}")

# エージェントを作成
agent = TDMPC(cfg)

# 学習済みモデルをロード
if cfg.pretrained_model_path:
    print(f"Loading model from {cfg.pretrained_model_path}")
    agent.load(cfg.pretrained_model_path)
else:
    print("Warning: No pretrained_model_path provided. Running with random weights.")

# エージェントを実行して可視化
obs = env.reset()
frames = []
done = False
t = 0
max_steps = cfg.episode_length

print("Starting rollout...")
while not done and t < max_steps:
    action = agent.act(obs, t0=(t == 0), eval_mode=True, step=0)
    
    # xarm環境などの辞書形式か、通常のnumpy配列かを判定して行動を適用
    act_np = action.cpu().numpy()
    obs, reward, done, info = env.step(act_np)
    
    # フレームをキャプチャ (envs.py の render 実装に従う)
    frame = env.render(mode='rgb_array', height=384, width=384)
    frames.append(frame)
    t += 1
    if t % 10 == 0:
        print(f"Step {t}/{max_steps}")

# 動画として保存
save_path = 'agent_video.mp4'
print(f"Saving video to {save_path}...")
imageio.mimsave(save_path, frames, fps=15)
print("Done!")