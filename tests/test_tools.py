"""
With this script we want to test the tools functions
"""

from activity_recognition.tools import loads_model, prediction, give_performance
import os
from pathlib import Path
from tensorflow.keras.models import Model
from pytest import fixture


def test_getcwd():
    dir = os.getcwd()
    assert dir == str(Path().absolute())


@fixture
def model():
    mdl_path = str(
        Path("activity_recognition/utils/single_ant_E,L,W,R,J_network.keras").absolute()
    )
    mdl = loads_model(mdl_path)
    return mdl



def test_tools(model):
    assert isinstance(model, Model)


def test_prediction(model):
    data = np
    y_pred = prediction(
        model, data
    )

