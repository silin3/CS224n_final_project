from evaluation import test_sonnet
from pathlib import Path

gold_path = "data/TRUE_sonnets_held_out_dev.txt"

test_files = [
    "predictions/sonnets-sonnet-dataaug-heldoutdev_v1.txt",
    "predictions/sonnets-sonnet-contemporary-heldoutdev_v1.txt",
    "predictions/sonnets-sonnet-prefix-heldoutdev_v1.txt",
    "predictions/sonnets-gpt2-10-1e-05-lora-all_attn-r8-a8_v1.txt",
    "predictions/sonnets-gpt2-10-1e-05-lora-attn_mlp-r8-a8_v1.txt",
    "predictions/sonnets-gpt2-10-1e-05-lora-qv-r8-a8_v1.txt",
]

output_path = "predictions/sonnet_eval_results.txt"

results = []

for test_path in test_files:
    try:
        score = test_sonnet(
            test_path=test_path,
            gold_path=gold_path
        )
        results.append((test_path, score))
        print(f"{test_path}: CHRF = {score:.6f}")
    except Exception as e:
        results.append((test_path, f"ERROR: {e}"))
        print(f"{test_path}: ERROR: {e}")

# 保存到 txt
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Gold file: {gold_path}\n\n")
    for test_path, score in results:
        if isinstance(score, float):
            f.write(f"{test_path}: CHRF = {score:.6f}\n")
        else:
            f.write(f"{test_path}: {score}\n")

print(f"\nResults saved to {output_path}")