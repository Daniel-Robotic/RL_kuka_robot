import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from kuka_env import KukaEnv


# def make_env():
#     env = KukaEnv(renders=False,
#                   is_enable_self_collision=True,
#                   max_distance=0.1,
#                   max_steps=3000)
#     return env
#
#
# # Создаем несколько параллельных сред
# num_envs = 4
# vec_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
#
# # Папка для сохранения чекпоинтов и лучшей модели
# log_dir = "./checkpoints/"
# os.makedirs(log_dir, exist_ok=True)
#
# # Callback для сохранения чекпоинтов
# checkpoint_callback = CheckpointCallback(
#     save_freq=1_000_000,  # Сохранять каждые n шагов
#     save_path=log_dir,
#     name_prefix="ppo_kuka_checkpoint"
# )
#
# # Callback для оценки и сохранения лучшей модели
# eval_callback = EvalCallback(
#     vec_env,
#     best_model_save_path=log_dir,  # Путь для сохранения лучшей модели
#     log_path=log_dir,              # Путь для логов оценки
#     eval_freq=5_000_000,           # Оценивать каждые n шагов
#     deterministic=True,            # Использовать детерминированные действия
#     render=False                   # Не рендерить во время оценки
# )
#
# # Настройка гиперпараметров
# hyperparams = {
#     "learning_rate": 1e-4,  # Скорость обучения
#     "n_steps": 4096,        # Количество шагов на обновление
#     "batch_size": 1024,     # Размер батча
# }
#
# # Инициализация модели PPO
# model = PPO(
#     "MlpPolicy",
#     vec_env,
#     verbose=1,
#     tensorboard_log="./logs",
#     # device="cpu",
#     **hyperparams
# )
#
# # Обучение модели
# total_timesteps = int(1e7)
# model.learn(
#     total_timesteps=total_timesteps,
#     # callback=[checkpoint_callback, eval_callback],
#     progress_bar=True
# )
#
# # Сохранение финальной модели
# model.save("ppo_kuka_final")

model = PPO.load("ppo_kuka_final")

# Тестирование модели
print("Тестирование модели...")
env = KukaEnv(renders=True, is_enable_self_collision=True)  # Создаем одну среду для тестирования
obs, _ = env.reset()

for _ in range(10):

    for i in range(2000):  # Запускаем агента на 1000 шагов
        start_time = time.time()  # Запускаем таймер
        # action = env.action_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        end_time = time.time()  # Останавливаем таймер

        # print("Total_time:", end_time - start_time)

        # print(action)
        # print(_states)
        # action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
            break

env.close()