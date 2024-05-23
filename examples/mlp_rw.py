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
from qlib.contrib.model.pytorch_nn import DNNModelPytorch
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
    model = DNNModelPytorch(batch_size=1000, lr=0.002,weight_decay = 0.0002, max_steps = 8000, optimizer = 'adam', pt_model_kwargs = {'input_dim': 158})

    model.fit(dataset, reweighter=reweighter)
    preds = model.predict(dataset)
    final_preds.append(preds)

final_result = pd.concat(final_preds, axis = 0)
final_result.to_pickle('./mlp_results_s20_axis0.pkl')

final_result = pd.concat(final_preds, axis = 1)
final_result.to_pickle('./mlp_results_s20_axis1.pkl')