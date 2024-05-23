import qlib
import pandas as pd
import sys, site
from pathlib import Path
from qlib.constant import REG_US
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.dataset.handler import DataHandlerLP

from qlib.model.utils import ConcatDataset
from qlib.data.dataset.weight import Reweighter
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from qlib.contrib.model.pytorch_transformer_rw import TransformerModel
# from qlib.contrib.model.pytorch_lstm import LSTM
from tqdm import tqdm

provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)

market = "SP500"
benchmark = "^gspc"

tasks = pd.read_pickle('/16t-2/cl/project/qlib/examples/benchmarks_dynamic/DDG-DA/tasks_s20.pkl')

final_preds = []
for task in tqdm(tasks, desc = 'tasks'):
    dataset = init_instance_by_config(task["dataset"])
    reweighter = task['reweighter']
    model = TransformerModel(d_feat = 158, seed = 0, n_jobs = 20, early_stop = 80)
    model.fit(dataset, reweighter=reweighter)
    preds = model.predict(dataset)
    final_preds.append(preds)

final_result = pd.concat(final_preds, axis = 0)
final_result.to_pickle('./transformer_results_s20_axis0.pkl')