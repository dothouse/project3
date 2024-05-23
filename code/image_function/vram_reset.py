import gc
from tensorflow.keras.backend import clear_session

def reset_vram():
    """Function to reset VRAM in Google Colab."""
    clear_session()
    gc.collect()
