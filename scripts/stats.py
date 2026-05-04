from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from mlcroscopy.data.dataset import OnlyExpt, ScopyTensor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    root = '../data'
    split = 'train.idx'

    # data = OnlyExpt(root, split)
    data = ScopyTensor(root=root, split_fptr=split)
    data[0]
    raise Exception
    loader = DataLoader(data, len(data))

    for x, _ in loader:
        # print(x[0].shape)
        print(x[0].mean(dim=[0, 2, 3]))
        print(x[0].std(dim=[0, 2, 3]))

