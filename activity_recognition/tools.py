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


def loads_model(mdl_path):
    return load_model(mdl_path)


def prediction(mdl, data, batch_size):
    
    #pred_val = mdl.predict(data, step=)
    
    #return pred_val
    pass


def give_performance():
    pass
