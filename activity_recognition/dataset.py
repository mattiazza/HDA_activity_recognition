import pickle
import shutil
from pathlib import Path

import numpy as np
from utils.dataset_utility import create_windows_antennas


def create_dataset(
    dir: Path,
    list_subdir: set[str],
    window_length: int,
    stride_length: int,
    labels_activities: list[str],
):
    activities = ",".join(labels_activities)

    for subdir in list_subdir:
        exp_dir = dir / subdir

        path_complete = exp_dir / f"complete_antennas_{activities}"
        path_train = exp_dir / f"train_antennas_{activities}"
        path_val = exp_dir / f"val_antennas_{activities}"
        path_test = exp_dir / f"test_antennas_{activities}"

        for pat in [path_train, path_val, path_test, path_complete]:
            shutil.rmtree(pat, ignore_errors=True)

        names = sorted(exp_dir.glob("S*"))

        csi_matrices = []
        lengths = []

        #######
        for activity in labels_activities:
            records = [p for p in names if str(p)[4] == activity]
            buffer = []

            for r in records:
                with open(r, "rb") as fp:  # Unpickling
                    arr = pickle.load(fp)

                arr_norm = arr - np.mean(arr, axis=0, keepdims=True)  # Normalize
                buffer.append(arr_norm.T)  # Traspose

            csi_matrices.append(np.asarray(buffer))

        csi_matrices_wind, labels_wind = create_windows_antennas(
            csi_matrices,
            labels_activities,
            window_length,
            stride_length,
            remove_mean=False,
        )

        num_windows = np.floor(
            (np.asarray(lengths) - window_length) / stride_length + 1
        )

        # creare directory

        names_complete = []
        for ii in range(len(csi_matrices_wind)):
            name_file = path_complete / f"{ii}.pkl"
            names_complete.append(name_file)
            with open(name_file, "wb") as fp:  # Pickling
                pickle.dump(csi_matrices_wind[ii], fp)
        name_labels = exp_dir / f"labels_complete_antennas_{activities}.pkl"
        with open(name_labels, "wb") as fp:  # Pickling
            pickle.dump(labels_wind, fp)
        name_f = exp_dir / f"files_complete_antennas_{activities}.pkl"
        with open(name_f, "wb") as fp:  # Pickling
            pickle.dump(names_complete, fp)
        name_f = exp_dir / f"num_windows_complete_antennas_{activities}.pkl"
        with open(name_f, "wb") as fp:  # Pickling
            pickle.dump(num_windows, fp)
