from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from mlcroscopy.data.dataset import OnlyExpt
from torch.utils.data import DataLoader

if __name__ == '__main__':
    root = '../data'
    split = 'train.idx'

    data = OnlyExpt(root, split)
    loader = DataLoader(data, len(data))

    for x, _ in loader:
        print(x.mean(axis=0))
        print(x.std(axis=0))

