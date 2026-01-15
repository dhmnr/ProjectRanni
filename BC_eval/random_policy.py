
import time
import eldengym

env = eldengym.make("Margit-v0", launch_game=False, host="192.168.48.1:50051", save_file_name="ER0000.Margit-v0.sl2", save_file_dir="C:\\Users\\DM\\AppData\\Roaming\\EldenRing\\76561198217475593")

HeroLocalPosX = env.client.get_attribute("HeroLocalPosX")

observation, info = env.reset()

print("Starting episode...")
print(f"Observation keys: {observation.keys()}")
print(f"Frame shape: {observation['frame'].shape}")
print(f"Info: {info}")
print("-" * 60)

for i in range(10):
    # Multi-binary action: each element toggles a key
    # For simple demo, just sample random actions
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    # Access normalized HP from info
    player_hp = info.get("player_hp_normalized", 0) * 100
    boss_hp = info.get("boss_hp_normalized", 100) * 100

    print(
        f"Step {i:3d} | Player HP: {player_hp:5.1f}% | Boss HP: {boss_hp:5.1f}% | Reward: {reward:+6.3f}"
    )

    # Small delay to make output readable
    time.sleep(0.2)

    if terminated or truncated:
        print("\n" + "=" * 60)
        if info.get("boss_hp_normalized", 1.0) <= 0:
            print("🎉 BOSS DEFEATED!")
        else:
            print("💀 YOU DIED")
        print("=" * 60 + "\n")

        # Reset for next episode
        observation, info = env.reset()

env.close()
print("\nEnvironment closed.")