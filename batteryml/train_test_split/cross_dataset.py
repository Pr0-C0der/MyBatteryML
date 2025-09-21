from typing import List, Tuple

from batteryml.builders import TRAIN_TEST_SPLITTERS
from batteryml.train_test_split.base import BaseTrainTestSplitter


@TRAIN_TEST_SPLITTERS.register()
class CrossDatasetTrainTestSplitter(BaseTrainTestSplitter):
    """Split by dataset directories: train on one dataset, test on another.

    Usage in config:
      train_test_split:
        name: 'CrossDatasetTrainTestSplitter'
        cell_data_path:
          - 'data/preprocessed/TRAIN_DATASET_DIR'
          - 'data/preprocessed/TEST_DATASET_DIR'

    The first path is used for training files; the second for testing files.
    """

    def __init__(self, cell_data_path: List[str]):
        # Base builds a single concatenated _file_list; we need both lists
        if not isinstance(cell_data_path, list) or len(cell_data_path) < 2:
            raise ValueError('Provide two paths: [train_dir, test_dir]')
        self._train_list = []
        self._test_list = []

        # Build train list from first path
        BaseTrainTestSplitter.__init__(self, [cell_data_path[0]])
        self._train_list = list(self._file_list)

        # Build test list from second path
        BaseTrainTestSplitter.__init__(self, [cell_data_path[1]])
        self._test_list = list(self._file_list)

    def split(self) -> Tuple[List, List]:
        return self._train_list, self._test_list


