from torch import cuda, device
from app import App

if __name__ == "__main__":
    try:
        device_ = device(f"cuda{cuda.current_device()}" if cuda.is_available() else "cpu")
    except:
        device_ = device("cpu")
    app = App(device_)
    app()