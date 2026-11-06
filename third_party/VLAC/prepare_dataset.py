#!/usr/bin/env python3
"""
准备VLAC训练数据集
从LIBERO demo images创建pair-wise critic训练样本
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import random

# tract repo root (third_party/VLAC/prepare_dataset.py -> repo); defaults are anchored here.
REPO_ROOT = Path(__file__).resolve().parents[2]
VLAC_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = str(REPO_ROOT / "data" / "atm" / "atm_libero" / "libero_goal")
DEFAULT_OUTPUT_FILE = str(VLAC_DIR / "data" / "train_dataset.json")
# Pair-wise critic prompt used at inference by evo_vlac.utils.model_utils.GAC_model (template "v3").
VLAC_PAIRWISE_PROMPT_V3 = (
    "Image-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer "
    "to achieving task objectives compared to the first image. + score means the second image is closer, "
    "- score means the first image is closer\nResponse the relative progressing of target task follow <score>. "
    "The target task is: <task> {} </task> <score>"
)

def get_task_description(task_dir: str) -> str:
    """从任务目录名提取任务描述"""
    task_name = os.path.basename(task_dir.rstrip('/'))
    task_desc = task_name.replace('_', ' ').replace('demo', '').strip()
    return task_desc

def get_image_files(demo_dir: str) -> List[str]:
    """获取demo目录中的所有图片文件，按数字排序"""
    # Absolute paths so ms-swift can load the images from any working directory.
    image_files = [os.path.abspath(p) for p in glob.glob(os.path.join(demo_dir, "agentview_*.png"))]
    image_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return image_files

def validate_sample(sample: Dict) -> bool:
    """
    验证样本是否符合ms-swift的要求
    
    Args:
        sample: 样本字典
    
    Returns:
        bool: 是否有效
    """
    # 检查必需字段
    if "messages" not in sample:
        return False
    
    if "images" not in sample:
        return False
    
    # 检查messages不为空
    if not isinstance(sample["messages"], list) or len(sample["messages"]) == 0:
        return False
    
    # 检查每个message的格式
    for msg in sample["messages"]:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            return False
        if len(msg["content"].strip()) == 0:
            return False
    
    # 检查images不为空
    if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
        return False
    
    # 检查图片文件是否存在
    for img_path in sample["images"]:
        if not isinstance(img_path, str):
            return False
        if not os.path.exists(img_path):
            # 警告但不阻止（ms-swift可能会处理相对路径）
            pass
    
    return True

def create_pairwise_samples(image_files: List[str], task_description: str, 
                           use_progressive_score: bool = True) -> List[Dict]:
    """
    创建pair-wise训练样本
    
    Args:
        image_files: 图片文件列表（按时间顺序）
        task_description: 任务描述
        use_progressive_score: 是否使用渐进式评分（早期帧评分较低，后期较高）
    """
    samples = []
    total_frames = len(image_files)
    
    # 验证图片文件存在
    valid_image_files = []
    for img_file in image_files:
        if os.path.exists(img_file):
            valid_image_files.append(img_file)
        else:
            print(f"警告: 图片文件不存在，跳过: {img_file}")
    
    if len(valid_image_files) < 2:
        return samples
    
    for i in range(len(valid_image_files) - 1):
        img1 = valid_image_files[i]
        img2 = valid_image_files[i + 1]
        
        # 验证图片文件存在
        if not os.path.exists(img1) or not os.path.exists(img2):
            print(f"警告: 跳过无效图片对: {img1}, {img2}")
            continue
        
        # 构建提示词（参考VLAC的v3模板）
        # Must match the inference prompt (evo_vlac GAC_model prompt_templete["v3"]) so the
        # fine-tuned critic is queried with the same text it was trained on.
        prompt = VLAC_PAIRWISE_PROMPT_V3.format(task_description)
        
        # 确保prompt不为空
        if not prompt or len(prompt.strip()) == 0:
            print(f"警告: 提示词为空，跳过")
            continue
        
        # 生成评分
        # 方法1：使用渐进式评分（假设任务逐步完成）
        if use_progressive_score:
            # 根据在轨迹中的位置调整基础评分
            # 早期：较小进步（2-4），中期：中等进步（4-6），后期：较大进步（6-8）
            progress_ratio = (i + 1) / total_frames
            if progress_ratio < 0.3:
                base_score = 3.0 + random.uniform(-1.0, 1.0)  # 早期：2-4
            elif progress_ratio < 0.7:
                base_score = 5.0 + random.uniform(-1.0, 1.0)  # 中期：4-6
            else:
                base_score = 7.0 + random.uniform(-1.0, 1.0)  # 后期：6-8
            score = max(0.0, min(10.0, base_score))  # 限制在0-10范围
        else:
            # 方法2：固定正向评分
            score = 5.0
        
        # 格式化评分为字符串（保留1位小数）
        score_str = f"{score:.1f}"
        
        # 确保score_str不为空
        if not score_str or len(score_str.strip()) == 0:
            print(f"警告: 评分为空，跳过")
            continue
        
        # 构建训练样本（ms-swift格式，使用InternVL2的对话格式）
        # ms-swift期望使用 "messages" 字段，格式为 role/content 而不是 from/value
        sample = {
            "images": [img1, img2],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": score_str
                }
            ]
        }
        
        # 验证样本
        if not validate_sample(sample):
            print(f"警告: 样本验证失败，跳过")
            continue
        
        samples.append(sample)
    
    return samples

def create_progress_samples(image_files: List[str], task_description: str,
                            max_stride: int = 10, reverse_prob: float = 0.5) -> List[Dict]:
    """VLAC-style pair-wise progress samples from one successful demo.

    The label is the signed task-progress difference between the two frames in percent,
    ``round(100 * (j - i) / (T - 1))``, written the way VLAC answers ("+3", "-7"). With
    probability ``reverse_prob`` the pair is shown in reverse order (negative label), so the
    critic also learns to recognise regressions.
    """
    files = [f for f in image_files if os.path.exists(f)]
    T = len(files)
    samples = []
    if T < 2:
        return samples
    for i in range(T - 1):
        j = min(T - 1, i + random.randint(1, max_stride))
        delta = int(round(100.0 * (j - i) / (T - 1)))
        if random.random() < reverse_prob:
            a, b, label = j, i, f"-{delta}"
        else:
            a, b, label = i, j, f"+{delta}"
        sample = {
            "images": [files[a], files[b]],
            "messages": [
                {"role": "user", "content": VLAC_PAIRWISE_PROMPT_V3.format(task_description)},
                {"role": "assistant", "content": label},
            ],
        }
        if validate_sample(sample):
            samples.append(sample)
    return samples


def prepare_dataset(
    data_root: str = DEFAULT_DATA_ROOT,
    output_file: str = DEFAULT_OUTPUT_FILE,
    task_filter: str = None,
    train_ratio: float = 0.9,
    use_progressive_score: bool = True,
    label_mode: str = "progress",
):
    """
    准备训练数据集
    
    Args:
        data_root: LIBERO数据根目录
        output_file: 输出JSON文件路径
        task_filter: 如果指定，只处理包含该字符串的任务目录
        train_ratio: 训练集比例
    """
    random.seed(42)  # reproducible pair sampling / labels
    all_samples = []
    
    # 遍历所有任务目录
    task_dirs = [d for d in os.listdir(data_root) 
                 if os.path.isdir(os.path.join(data_root, d)) and d.endswith('_demo')]
    
    if task_filter:
        task_dirs = [d for d in task_dirs if task_filter in d]
    
    print(f"找到 {len(task_dirs)} 个任务目录")
    
    for task_dir in tqdm(task_dirs, desc="处理任务"):
        task_path = os.path.join(data_root, task_dir)
        images_dir = os.path.join(task_path, "images")
        
        if not os.path.exists(images_dir):
            print(f"警告: {images_dir} 不存在，跳过")
            continue
        
        task_description = get_task_description(task_dir)
        print(f"\n处理任务: {task_description}")
        
        # 获取所有demo目录
        demo_dirs = [d for d in os.listdir(images_dir) 
                     if os.path.isdir(os.path.join(images_dir, d)) and d.startswith('demo_')]
        demo_dirs.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        
        print(f"  找到 {len(demo_dirs)} 个demo")
        
        for demo_dir in tqdm(demo_dirs, desc=f"  处理 {task_description}", leave=False):
            demo_path = os.path.join(images_dir, demo_dir)
            image_files = get_image_files(demo_path)
            
            if len(image_files) < 2:
                continue
            
            # 创建pair-wise样本
            if label_mode == "progress":
                samples = create_progress_samples(image_files, task_description)
            else:  # "heuristic": original position-based random scores
                samples = create_pairwise_samples(
                    image_files,
                    task_description,
                    use_progressive_score=use_progressive_score
                )
            all_samples.extend(samples)
    
    print(f"\n总共创建了 {len(all_samples)} 个训练样本")
    
    # 验证所有样本
    print("验证样本格式...")
    valid_samples = []
    invalid_count = 0
    for i, sample in enumerate(all_samples):
        if validate_sample(sample):
            valid_samples.append(sample)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # 只打印前5个无效样本
                print(f"警告: 样本 {i} 验证失败: {sample.get('images', 'N/A')}")
    
    if invalid_count > 0:
        print(f"过滤了 {invalid_count} 个无效样本")
    
    if len(valid_samples) == 0:
        raise ValueError("错误: 没有有效的训练样本！请检查数据路径和格式。")
    
    print(f"有效样本数: {len(valid_samples)}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(valid_samples)
    
    # 分割训练集和验证集
    split_idx = int(len(valid_samples) * train_ratio)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]
    
    # 最终验证
    if len(train_samples) == 0:
        raise ValueError("错误: 训练集为空！")
    if len(val_samples) == 0:
        raise ValueError("错误: 验证集为空！")
    
    # 保存训练集
    train_output = output_file.replace('.json', '_train.json')
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    
    # 再次验证训练集样本
    final_train_samples = [s for s in train_samples if validate_sample(s)]
    if len(final_train_samples) != len(train_samples):
        print(f"警告: 训练集中有 {len(train_samples) - len(final_train_samples)} 个无效样本被过滤")
    
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(final_train_samples, f, indent=2, ensure_ascii=False)
    print(f"训练集已保存到: {train_output} ({len(final_train_samples)} 个样本)")
    
    # 验证保存的文件
    with open(train_output, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        print(f"验证: 已保存的训练集包含 {len(saved_data)} 个样本")
        if len(saved_data) > 0:
            print(f"验证: 第一个样本包含 {len(saved_data[0].get('messages', []))} 条消息")
    
    # 保存验证集
    val_output = output_file.replace('.json', '_val.json')
    
    # 再次验证验证集样本
    final_val_samples = [s for s in val_samples if validate_sample(s)]
    if len(final_val_samples) != len(val_samples):
        print(f"警告: 验证集中有 {len(val_samples) - len(final_val_samples)} 个无效样本被过滤")
    
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(final_val_samples, f, indent=2, ensure_ascii=False)
    print(f"验证集已保存到: {val_output} ({len(final_val_samples)} 个样本)")
    
    # 验证保存的文件
    with open(val_output, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        print(f"验证: 已保存的验证集包含 {len(saved_data)} 个样本")
        if len(saved_data) > 0:
            print(f"验证: 第一个样本包含 {len(saved_data[0].get('messages', []))} 条消息")
    
    return train_output, val_output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备VLAC训练数据集")
    parser.add_argument("--data_root", type=str, 
                       default=DEFAULT_DATA_ROOT,
                       help="LIBERO数据根目录")
    parser.add_argument("--output_file", type=str,
                       default=DEFAULT_OUTPUT_FILE,
                       help="输出JSON文件路径")
    parser.add_argument("--task_filter", type=str, default=None,
                       help="任务过滤（只处理包含该字符串的任务）")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="训练集比例")
    parser.add_argument("--label_mode", choices=["progress", "heuristic"], default="progress",
                       help="progress: VLAC式带符号进度差标签(含反向负样本)；heuristic: 原始按位置随机正分")
    parser.add_argument("--use_progressive_score", action="store_true", default=True,
                       help="使用渐进式评分（根据轨迹位置调整评分）")
    parser.add_argument("--no_progressive_score", dest="use_progressive_score", 
                       action="store_false",
                       help="不使用渐进式评分，使用固定评分")
    
    args = parser.parse_args()
    
    prepare_dataset(
        data_root=args.data_root,
        output_file=args.output_file,
        task_filter=args.task_filter,
        train_ratio=args.train_ratio,
        use_progressive_score=args.use_progressive_score,
        label_mode=args.label_mode,
    )
