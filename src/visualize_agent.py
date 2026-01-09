import torch
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
import numpy as np

# 設定を読み込む
cfg = parse_cfg(Path('cfgs'))
cfg.task = 'antmaze-medium-play-v2'  # タスクを指定

# 環境を作成（これにより cfg.obs_shape などが設定される）
env = make_env(cfg)

# デバッグ: cfgの状態を確認
print(f"DEBUG: cfg.modality = {cfg.modality}")
print(f"DEBUG: cfg.obs_shape = {cfg.obs_shape}")
print(f"DEBUG: type(cfg.obs_shape) = {type(cfg.obs_shape)}")

# エージェントを作成（環境作成後に cfg.obs_shape が必要）
agent = TDMPC(cfg)

# 学習済みモデルをロード
agent.load(f'logs/{cfg.task}/state/default/1/models/final.pt')

# エージェントを実行して可視化
obs = env.reset()
frames = []
done = False
t = 0

while not done:
    action = agent.act(obs, t0=(t == 0), eval_mode=True, step=0)
    obs, reward, done, info = env.step(action.cpu().numpy())
    
    # フレームをキャプチャ
    frame = env.render(mode='rgb_array', height=384, width=384)
    frames.append(frame)
    t += 1

# 動画として保存（例：imageio を使用）
import imageio
imageio.mimsave('agent_video.mp4', frames, fps=15)