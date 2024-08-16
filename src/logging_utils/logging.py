import psutil


def get_ram_used():
    return lambda: psutil.virtual_memory().used / (1024.0**3)


def get_ram_total():
    return lambda: psutil.virtual_memory().total / (1024.0**3)
