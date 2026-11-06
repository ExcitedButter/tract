"""Fine-tune VLAC (InternVL2-8B based pair-wise critic) with ms-swift 3.x.

Data: run ``prepare_dataset.py`` first; it writes ms-swift ``messages``/``images``
JSON files (``*_train.json`` / ``*_val.json``).

Example (single node, LoRA, quick smoke test):
    python third_party/VLAC/train.py --use_lora --max_steps 20 --yes
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# tract repo root (third_party/VLAC/train.py -> repo); default paths are anchored here.
REPO_ROOT = Path(__file__).resolve().parents[2]
VLAC_DIR = Path(__file__).resolve().parent
HF_MODEL_ID = "InternRobotics/VLAC-8b"
SYSTEM_PROMPT = (
    "You are a visual-language assistant designed to interpret spatial and task-related information "
    "from images and text. Provide precise, context-aware responses and actionable guidance to assist "
    "in achieving task objectives."
)


def main():
    parser = argparse.ArgumentParser(description="VLAC模型训练 (ms-swift 3.x)")
    parser.add_argument("--pretrained_model", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "VLAC-8b"),
                        help=f"预训练模型路径（本地不存在时从HuggingFace下载 {HF_MODEL_ID}）")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=str(VLAC_DIR / "models" / "1"),
                        help="checkpoint保存目录")
    parser.add_argument("--train_json", type=str,
                        default=str(VLAC_DIR / "data" / "train_dataset_train.json"),
                        help="训练数据集JSON文件（messages格式）")
    parser.add_argument("--val_json", type=str,
                        default=str(VLAC_DIR / "data" / "train_dataset_val.json"),
                        help="验证数据集JSON文件（messages格式）")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=None, help="最多训练步数（覆盖 num_epochs，用于快速测试）")
    parser.add_argument("--batch_size", type=int, default=2, help="每设备batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--use_lora", action="store_true", help="使用LoRA微调（节省显存）")
    parser.add_argument("--freeze_vit", action=argparse.BooleanOptionalAction, default=True, help="冻结视觉编码器和aligner")
    parser.add_argument("--attn_impl", type=str, default=None, choices=["flash_attn", "sdpa", "eager"],
                        help="注意力实现（未安装 flash_attn 时用 eager / sdpa）")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--yes", action="store_true", help="不询问确认，直接开始训练")
    args = parser.parse_args()

    # 检查数据集是否存在
    for name, path in (("训练", args.train_json), ("验证", args.val_json)):
        if not os.path.exists(path):
            print(f"错误: {name}数据集不存在: {path}")
            print("请先运行 prepare_dataset.py 准备数据集")
            sys.exit(1)

    # 模型：本地 VLAC-8b 权重优先，否则从 HuggingFace 下载
    if os.path.isdir(args.pretrained_model):
        model = args.pretrained_model
        use_hf = []
        print(f"使用本地预训练模型: {model}")
        if not os.path.exists(os.path.join(model, "config.json")):
            print(f"警告: 配置文件不存在: {os.path.join(model, 'config.json')}")
    else:
        print(f"警告: 预训练模型不存在: {args.pretrained_model}")
        print(f"将使用HuggingFace上的模型: {HF_MODEL_ID}")
        model = HF_MODEL_ID
        use_hf = ["--use_hf", "true"]

    # ms-swift 3.x sft: --model 决定加载的权重；数据集直接传 messages/images JSON 文件
    cmd_parts = [
        "swift", "sft",
        "--model", model,
        "--model_type", "internvl2",
        *use_hf,
        "--train_type", "lora" if args.use_lora else "full",
        "--dataset", args.train_json,
        "--val_dataset", args.val_json,
        "--system", SYSTEM_PROMPT,
        "--max_length", "10240",
        "--output_dir", args.checkpoint_dir,
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", "0.1",
        "--num_train_epochs", str(args.num_epochs),
        "--lr_scheduler_type", "cosine",
        "--warmup_steps", "100",
        "--save_steps", str(args.save_steps),
        "--eval_steps", str(args.eval_steps),
        "--logging_steps", "10",
        "--save_total_limit", "3",
        "--dataloader_num_workers", "4",
        "--seed", "42",
        "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true",
        "--freeze_vit", str(args.freeze_vit).lower(),
        "--freeze_aligner", str(args.freeze_vit).lower(),
    ]
    if args.max_steps is not None:
        cmd_parts.extend(["--max_steps", str(args.max_steps)])
    if args.attn_impl:
        cmd_parts.extend(["--attn_impl", args.attn_impl])
    if args.use_lora:
        cmd_parts.extend([
            "--lora_rank", "8",
            "--lora_alpha", "32",
            "--lora_dropout", "0.05",
            "--target_modules", "all-linear",
        ])

    print("\n" + "=" * 80)
    print("训练配置:")
    print("=" * 80)
    print(f"预训练模型: {model}")
    print(f"训练数据集: {args.train_json}")
    print(f"验证数据集: {args.val_json}")
    print(f"Checkpoint目录: {args.checkpoint_dir}")
    print(f"训练轮数: {args.num_epochs}" + (f" (max_steps={args.max_steps})" if args.max_steps else ""))
    print(f"Batch size: {args.batch_size}")
    print(f"梯度累积: {args.gradient_accumulation_steps}")
    print(f"学习率: {args.learning_rate}")
    print(f"使用LoRA: {args.use_lora}")
    print(f"冻结ViT: {args.freeze_vit}")
    print("=" * 80)
    print("\n训练命令:")
    print(" ".join(cmd_parts))
    print("=" * 80 + "\n")

    # 确认（--yes 或非交互环境时跳过）
    if not args.yes and sys.stdin is not None and sys.stdin.isatty():
        response = input("是否开始训练? (y/n): ")
        if response.lower() != "y":
            print("训练已取消")
            sys.exit(0)

    print("开始训练...")
    try:
        subprocess.run(cmd_parts, check=True)
        print(f"\n训练完成！模型已保存到: {args.checkpoint_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)


if __name__ == "__main__":
    main()
