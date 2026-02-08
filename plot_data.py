import pandas as pd
import matplotlib.pyplot as plt

# ログファイルを読み込む
task_name = 'xarm_lift'
modality = 'all'
df = pd.read_csv(f'logs/{task_name}/{modality}/default/1/eval.log')

# 学習曲線をプロット
plt.plot(df['env_step'], df['episode_reward'])
plt.xlabel('Environment Steps [1e6]')
plt.ylabel('Episode Reward')
plt.title('Training Curve')
plt.show()