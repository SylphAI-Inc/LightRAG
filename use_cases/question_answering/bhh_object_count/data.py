"""Prepare classes, components, functions that will prepare and post processing the data"""

from dataclasses import dataclass, field


from lightrag.core import DataClass, fun_to_component
from lightrag.datasets.big_bench_hard import BigBenchHard
from lightrag.utils.data import subset_dataset


@dataclass
class ObjectCountPredData(DataClass):
    """Dataclass for structed prediction"""

    thought: str = field(metadata={"desc": "List your step by step reasoning"})
    answer: int = field(
        metadata={"desc": "The answer to the question, only numerical values"}
    )


def _parse_integer_answer(answer: str, only_first_line: bool = False):
    """A function to component that will parse the answer from a string. Used for string output"""
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]
        answer = answer.strip()
        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][
            -1
        ]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        answer = 0

    return answer


@fun_to_component
def parse_integer_answer(answer: str, only_first_line: bool = False):
    return _parse_integer_answer(answer, only_first_line)


def load_datasets(max_samples: int = None, dataset_path: str = "cache_datasets"):
    """Load the dataset"""
    dataset_name = "BBH_object_counting"
    train_data = BigBenchHard(dataset_name, split="train", root=dataset_path)
    val_data = BigBenchHard(dataset_name, split="val", root=dataset_path)
    test_data = BigBenchHard(dataset_name, split="test", root=dataset_path)

    # Limit the number of samples
    if max_samples:
        train_data = subset_dataset(train_data, max_samples)
        val_data = subset_dataset(val_data, max_samples)
        test_data = subset_dataset(test_data, max_samples)

    return train_data, val_data, test_data