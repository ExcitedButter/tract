"""Pair-wise VLAC critic on an image sequence.

Input n images, output a critic_list of length n-1 and a value_list of length n. The critic scores
adjacent images (i, i+1): positive if i+1 is closer to accomplishing the task than i, negative
otherwise; value_list accumulates the critic into task progress (0-100).

    python third_party/VLAC/evo_vlac/examples/image_pair.py [--model checkpoints/VLAC-8b] [--no-ref]
"""

import argparse
from pathlib import Path
import sys

from PIL import Image

HERE = Path(__file__).resolve().parent
VLAC_DIR = HERE.parents[1]  # third_party/VLAC
REPO_ROOT = HERE.parents[3]  # tract repo root
sys.path.insert(0, str(VLAC_DIR))

from evo_vlac import GAC_model  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default=str(REPO_ROOT / "checkpoints" / "VLAC-8b"), help="VLAC-8b weights")
parser.add_argument("--no-ref", action="store_true", help="do not condition on the reference trajectory")
args = parser.parse_args()

test_images = [Image.open(HERE / "images" / "test" / f"{i}.png").convert("RGB") for i in range(8)]
# (optional) up to 11 reference-trajectory images; they help adaptation to new tasks / scenes.
ref_images = [Image.open(HERE / "images" / "ref" / f"{i}.png").convert("RGB") for i in range(11)]
task_description = "Close the bottom drawer of the cabinet demo."

critic = GAC_model(tag="critic")
critic.init_model(model_path=args.model, model_type="internvl2", device_map="cuda:0")
critic.temperature = 0.5
critic.top_k = 1
critic.set_config()
critic.set_system_prompt()

critic_list, value_list = critic.get_trajectory_critic(
    task=task_description,
    image_list=test_images,
    ref_image_list=None if args.no_ref else ref_images,
    batch_num=5,  # max batch size when generating critic outputs
    ref_num=0 if args.no_ref else len(ref_images),
    rich=False,  # True: decimal values
    reverse_eval=False,  # True: reverse the evaluation (VROC)
)

print("=" * 80)
print("value_list: ", value_list)
print("critic_list:", critic_list)
