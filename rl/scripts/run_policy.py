import sys
from stable_baselines3 import SAC
from rl.env.pepper_nav_env import PepperNavEnv
from rl.env.routes import load_routes

MODELS = {"standard": "rl/runs/20261007_105842_standard_seed1/final_model.zip",
          "photogrammetry": "rl/runs/20261007_105847_photogrammetry_seed0/final_model.zip"}
WORLDS = {"standard": "room_standard", "photogrammetry": "room_photogrammetry"}

branch = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0

routes = load_routes("final_scan", variant="v2")[first:]
env = PepperNavEnv(world_name=WORLDS[branch], branch=branch, source="real",
                   randomiser=None, seed=0, room="final_scan", pose_source="odometry")
model = SAC.load(MODELS[branch], env=env)
try:
    for route in routes:
        obs, _ = env.reset(options={"route": route})
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print(f"{info['route_id'] or 'unnamed'}: "
              f"{'reached' if info['reached_goal'] else 'failed'} "
              f"d={info['distance_to_goal_m']:.2f} l={info['path_length_m']:.2f} "
              f"l*={info['optimal_length_m'] or 0.0:.2f} {info['collisions']} collisions "
              f"{info['duration_s']:.1f}s{' TIMEOUT' if info['timed_out'] else ''}")
finally:
    env.ros.stop()      # always stop the wheels, even after a crash or Ctrl-C
    env.ros.spin(0.2)
env.close()