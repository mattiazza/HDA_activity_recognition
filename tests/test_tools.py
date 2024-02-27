"""
With this script we want to test the tools functions
"""

from activity_recognition.tools import loads_model, prediction, give_performance
import os
from pathlib import Path
from tensorflow.keras.models import Model


def test_getcwd():
    dir = os.getcwd()
    assert dir == str(Path().absolute())


def test_loads_model():
    mdl_path = str(
        Path("activity_recognition/utils/single_ant_E,L,W,R,J_network.keras").absolute()
    )
    print(f"mdl_path: \n\n{mdl_path}\n")
    loaded_model = loads_model(mdl_path)

    assert isinstance(loaded_model, Model)
