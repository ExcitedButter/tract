import abc
import os
import glob
import random
import re
import torch

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    init_states_folder: str = ""


# LIBERO-plus / Pro splits: language + scene edits live in per-suite BDDL, not filenames.
_PRO_VARIANT_SUITE_RE = re.compile(r"_(?:lan|object|swap|task)$")


def _bddl_language_instruction(suite_name: str, stem: str) -> str | None:
    """Read ``(:language ...)`` from ``bddl_files/<suite>/<stem>.bddl``."""
    lang_bddl = os.path.join(get_libero_path("bddl_files"), suite_name, f"{stem}.bddl")
    if not os.path.isfile(lang_bddl):
        return None
    import importlib

    bddl_utils = importlib.import_module("libero.libero.envs.bddl_utils")
    text = bddl_utils.get_problem_info(lang_bddl)["language_instruction"]
    return text.strip() if text else None


def _language_from_filename_stem(stem: str) -> str:
    """Legacy fallback: derive instruction text from task filename (pre-Pro suites)."""
    if stem[0].isupper():  # LIBERO-100 / LIBERO-10 style
        if "SCENE10" in stem:
            language = " ".join(stem[stem.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(stem[stem.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(stem.split("_"))
    return language.strip()


def grab_language_from_filename(x, *, suite_name: str | None = None):
    """Resolve task language for policy prompts.

    All ``*_lan``, ``*_object``, ``*_swap``, ``*_task`` suites (and any suite with a
    matching BDDL under ``bddl_files/<suite>/``) use ``(:language ...)`` from BDDL.
    Filename parsing is only used when no suite BDDL is available (e.g. legacy paths).
    """
    en_ext = x.find(".bddl")
    stem = x[:en_ext] if en_ext != -1 else x
    if "__libero_" in stem:
        stem = stem.split("__libero_", 1)[0]

    if suite_name:
        bddl_lang = _bddl_language_instruction(suite_name, stem)
        if bddl_lang:
            return bddl_lang
        _bddl_required = (
            _PRO_VARIANT_SUITE_RE.search(suite_name) is not None
            or suite_name
            in (
                "libero_background",
                "libero_robot",
                "libero_viewpoint",
            )
        )
        if _bddl_required:
            raise FileNotFoundError(
                f"Missing or empty (:language ...) in bddl_files/{suite_name}/{stem}.bddl. "
                f"Pro-derived suites must use BDDL prompts, not filename parsing."
            )

    return _language_from_filename_stem(stem)


libero_suites = [
    "libero_spatial",
    "libero_spatial_cross",
    "libero_spatial_sawyer",
    "libero_spatial_lan",
    "libero_spatial_object",
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_object",
    "libero_object_cross",
    "libero_object_sawyer",
    "libero_object_lan",
    "libero_object_object",
    "libero_object_swap",
    "libero_object_task",
    "libero_goal",
    "libero_goal_cross",
    "libero_goal_sawyer",
    "libero_goal_lan",
    "libero_goal_object",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_90",
    "libero_10",
    "libero_10_arm",
    "libero_object_arm",
    "libero_10_cross",
    "libero_10_sawyer",
    "libero_10_object",
    "libero_10_lan",
    "libero_10_swap",
    "libero_10_task",
    "libero_10_benchmark",
    "libero_integral",
    "libero_background",
    "libero_robot",
    "libero_viewpoint",
]

# Suites that ship parallel `{task}.init` copies (same tensor as `.pruned_init`).
_LIBERO_SUITE_INIT_STATE_SUFFIX = {
    "libero_spatial_lan": "init",
    "libero_spatial_object": "init",
    "libero_spatial_swap": "init",
    "libero_spatial_task": "init",
    "libero_object_lan": "init",
    "libero_object_object": "init",
    "libero_object_swap": "init",
    "libero_object_task": "init",
    "libero_goal_lan": "init",
    "libero_goal_object": "init",
    "libero_goal_swap": "init",
    "libero_goal_task": "init",
    "libero_10_object": "init",
    "libero_10_lan": "init",
    "libero_10_swap": "init",
    "libero_10_task": "init",
}

task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}
    init_suffix = _LIBERO_SUITE_INIT_STATE_SUFFIX.get(libero_suite, "pruned_init")

    for slot_i, item in enumerate(libero_task_map[libero_suite]):
        init_states_folder = ""
        if libero_suite == "libero_integral":
            if len(item) == 3:
                problem_folder, task, init_states_folder = item
            else:
                problem_folder, task = item
                init_states_folder = ""
            init_src_suffix = _LIBERO_SUITE_INIT_STATE_SUFFIX.get(
                init_states_folder or problem_folder, "pruned_init"
            )
            map_key = f"integral_{slot_i:02d}"
            if "_view_" in task:
                init_states_file = f"{task.split('_view_')[0]}.{init_src_suffix}"
            else:
                init_states_file = f"{task}.{init_src_suffix}"
            language = grab_language_from_filename(
                task + ".bddl", suite_name=problem_folder
            )
            task_maps[libero_suite][map_key] = Task(
                name=task,
                language=language,
                problem="Libero",
                problem_folder=problem_folder,
                bddl_file=f"{task}.bddl",
                init_states_file=init_states_file,
                init_states_folder=init_states_folder,
            )
            continue
        elif libero_suite in ("libero_viewpoint", "libero_robot"):
            init_states_folder, task = item
            problem_folder = libero_suite
            init_src_suffix = _LIBERO_SUITE_INIT_STATE_SUFFIX.get(
                init_states_folder, "pruned_init"
            )
            base_task = task.split("_view_")[0]
            map_key = task
            task_maps[libero_suite][map_key] = Task(
                name=task,
                language=grab_language_from_filename(
                    task + ".bddl", suite_name=problem_folder
                ),
                problem="Libero",
                problem_folder=problem_folder,
                bddl_file=f"{task}.bddl",
                init_states_file=f"{base_task}.{init_src_suffix}",
                init_states_folder=init_states_folder,
            )
            continue
        else:
            task = item
            problem_folder = libero_suite
            init_src_suffix = init_suffix
            map_key = task

        language = grab_language_from_filename(
            task + ".bddl", suite_name=problem_folder
        )
        task_maps[libero_suite][map_key] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=problem_folder,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.{init_src_suffix}",
            init_states_folder=init_states_folder,
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if self.name in (
            "libero_90",
            "libero_integral",
            "libero_background",
            "libero_robot",
            "libero_viewpoint",
        ):
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def _resolve_init_states_path(self, task: Task) -> str:
        init_folder = task.init_states_folder or task.problem_folder
        init_root = get_libero_path("init_states")
        candidates = [task.init_states_file]
        if "_view_" in task.init_states_file:
            candidates.append(
                task.init_states_file.split("_view_")[0]
                + "."
                + task.init_states_file.split(".")[-1]
            )
        if "_table_" in task.init_states_file:
            candidates.append(re.sub(r"_table_\d+", "", task.init_states_file))
        if "_tb_" in task.init_states_file:
            candidates.append(re.sub(r"_tb_\d+", "", task.init_states_file))
        seen = set()
        for init_file in candidates:
            if init_file in seen:
                continue
            seen.add(init_file)
            path = os.path.join(init_root, init_folder, init_file)
            if os.path.isfile(path):
                return path
        return os.path.join(init_root, init_folder, task.init_states_file)

    def get_task_init_states(self, i):
        init_states_path = self._resolve_init_states_path(self.tasks[i])
        # PyTorch 2.6+ defaults weights_only=True; LIBERO init files contain pickled numpy objects.
        init_states = torch.load(init_states_path, weights_only=False)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_LAN(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_lan"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_SWAP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_swap"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_TASK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_task"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_LAN(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_lan"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_SWAP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_swap"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_TASK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_task"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_LAN(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_lan"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_SWAP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_swap"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_TASK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_task"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_ARM(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_arm"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_ARM(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_arm"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_CROSS(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_cross"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_SAWYER(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_sawyer"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_CROSS(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_cross"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_SAWYER(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_sawyer"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_CROSS(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_cross"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_SAWYER(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_sawyer"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_CROSS(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_cross"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_SAWYER(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_sawyer"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_BENCHMARK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_benchmark"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_LAN(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_lan"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_SWAP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_swap"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_TASK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_task"
        self._make_benchmark()


@register_benchmark
class LIBERO_INTEGRAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_integral"
        self._make_benchmark()


@register_benchmark
class LIBERO_BACKGROUND(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_background"
        self._make_benchmark()


@register_benchmark
class LIBERO_ROBOT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_robot"
        self._make_benchmark()


@register_benchmark
class LIBERO_VIEWPOINT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_viewpoint"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()
