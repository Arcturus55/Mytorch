import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(attr, value):
    old_value = getattr(Config, attr)
    setattr(Config, attr, value)
    try:
        yield
    finally:
        setattr(Config, attr, old_value)

def no_grad():
    return using_config("enable_backprop", False)