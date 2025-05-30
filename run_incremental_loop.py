import os
import subprocess
from pathlib import Path


def run_incremental_sequence(task_dir="data", base_ckpt="hanet_base.ckpt"):
    """
    Loop through all incremental tasks and train HANet incrementally.
    Deletes previous checkpoint after each task to save space.

    Args:
        task_dir (str): Path to directory containing incremental_task_*.jsonl
        base_ckpt (str): Path to the base checkpoint to start with
    """
    task_files = sorted(Path(task_dir).glob("incremental_task_*.jsonl"))[:2]
    model_ckpt = base_ckpt

    for i, task_file in enumerate(task_files):
        print(f"\n🚀 Running Incremental Task {i+1}: {task_file.name}")

        output_ckpt = f"hanet_inc_{i+1}.ckpt"
        cmd = [
            "python", "train_incremental.py",
            "--task", str(task_file),
            "--base_ckpt", model_ckpt,
            "--output", output_ckpt
        ]
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Remove previous checkpoint (if not the initial base)
        if model_ckpt != base_ckpt and os.path.exists(model_ckpt):
            print(f"🗑️ Deleting old checkpoint: {model_ckpt}")
            os.remove(model_ckpt)

        model_ckpt = output_ckpt  # Update for next step

    print("\n✅ All incremental tasks finished.")


if __name__ == "__main__":
    run_incremental_sequence()
