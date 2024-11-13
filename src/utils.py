from functools import partial
from tqdm import tqdm

# can't get tqdm.notebook to work, this also works
tqdm = partial(tqdm, position=0, ncols=100)