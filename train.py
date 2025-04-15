import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from kuka_env import KukaEnv

# def make_env():
#     env = KukaEnv(renders=False, is_enable_self_collision=True)
#     env = Monitor(env)
#     return env
#
# # Создаем несколько параллельных сред (например, 4)
# num_envs = 4
# vec_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)
#
# # Инициализируем модель PPO
# model = PPO(
#     "MultiInputPolicy",  # Policy для словарного пространства наблюдений
#     vec_env,
#     verbose=1,
#     tensorboard_log="./ppo_kuka_tensorboard/"  # Опционально: логи TensorBoard
# )
#
# # Обучаем модель
# total_timesteps = 1_000_000  # Общее количество временных шагов
# model.learn(total_timesteps=total_timesteps,
#             progress_bar=True)
#
# # Сохраняем модель
# model.save("ppo_kuka")

model = PPO.load("ppo_kuka")

# Тестирование модели
print("Тестирование модели...")
env = KukaEnv(renders=True, is_enable_self_collision=True)  # Создаем одну среду для тестирования
obs, _ = env.reset()

for i in range(1000):  # Запускаем агента на 1000 шагов
    action, _states = model.predict(obs, deterministic=True)
    # print(action)
    # print(_states)
    # action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()