import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
import time
import datetime


# for i in trange(100):
with tqdm(range(100)) as t:
    for i in t:
        remaining = (t.total - t.n- 1)/ t.format_dict["rate"] if t.format_dict["rate"] and t.total else 0  # Seconds
        remaining = str(datetime.timedelta(seconds=round(remaining))) #Hour:min:sec



