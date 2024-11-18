import subprocess

for i in range(5):
    print(f"Executing main.py (Run {i + 1}/5)...")
    subprocess.run(["python", "main.py", "--name", f"{i}st", "--seed", f"{i}"])