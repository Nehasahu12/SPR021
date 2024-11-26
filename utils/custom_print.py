import random
xprint = print


def print(*args, **kwargs):
    color_code = f"\033[38;5;{random.randint(150, 255)}m"
    # xprint(color_code, end="")
    xprint(*args, **kwargs)
    # xprint("\033[0m", end="")
