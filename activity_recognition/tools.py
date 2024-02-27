####################
#      TOOLS       #
####################
"""
• loads_model()
• prediction()
• give_performance()
"""

### Import space

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.saving import load_model
from tensorflow.keras.model import Model


def loads_model(mdl_path: str) -> Model:
    return load_model(mdl_path)


def prediction(mdl, data, batch_size):
    pred_val = mdl.predict(data, step=batch_size)
    return pred_val


def give_performance():
    pass
