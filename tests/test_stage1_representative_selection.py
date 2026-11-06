import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_representative_uses_median_top1_and_id_tiebreak_not_cosine_distance():
    df = pd.DataFrame({"fold":[91,144,83,1], "top1":[.8,.1,.9,1.],
                       "intra_distance_mean":[.1,.2,.8,.9],
                       "nearest_inter_distance_mean":[.4,.5,.7,.6],
                       "margin_mean":[.3,-.4,-.1,.4]})
    for name in ["notebooks/01_stage1_retrieval.ipynb"]:
        nb = json.loads((ROOT/name).read_text())
        scope = {"df":df.copy(),"np":np,"pd":pd}
        exec(compile("".join(nb["cells"][5]["source"]),name,"exec"),scope)
        assert np.isclose(scope["median_accuracy"], .85)
        assert int(scope["representative"]["fold"]) == 83
        assert int(scope["lowest_accuracy"]["fold"]) == 144
        assert int(df.loc[df["intra_distance_mean"].idxmax(), "fold"]) == 1
        assert "glob('*.png')" not in "".join(nb["cells"][9]["source"])


def test_minimum_top1_ties_use_smallest_id_independent_of_row_order():
    df = pd.DataFrame({"fold":[144,91,83,5], "top1":[.1,.8,.9,.1],
                       "intra_distance_mean":[.2,.1,.3,.9],
                       "nearest_inter_distance_mean":[.4,.5,.7,.6],
                       "margin_mean":[.3,-.4,-.1,.4]})
    nb = json.loads((ROOT/"notebooks/01_stage1_retrieval.ipynb").read_text())
    for frame in [df, df.iloc[::-1]]:
        scope = {"df":frame.copy(),"np":np,"pd":pd}
        exec(compile("".join(nb["cells"][5]["source"]),"selection","exec"),scope)
        assert int(scope["lowest_accuracy"]["fold"]) == 5
