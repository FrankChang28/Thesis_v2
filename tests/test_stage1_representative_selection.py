import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_representative_uses_median_top1_and_id_tiebreak_not_cosine_distance():
    df = pd.DataFrame({"fold":[91,144,83,1], "top1":[.8,.1,.9,1.],
                       "intra_distance_mean":[.1,.9,.8,.2],
                       "nearest_inter_distance_mean":[.4,.5,.7,.6],
                       "margin_mean":[.3,-.4,-.1,.4]})
    for name in ["notebooks/01_stage1_retrieval.ipynb"]:
        nb = json.loads((ROOT/name).read_text())
        scope = {"df":df.copy(),"np":np,"pd":pd}
        exec(compile("".join(nb["cells"][5]["source"]),name,"exec"),scope)
        assert scope["median_accuracy"] == pytest.approx(.85)
        assert int(scope["representative"]["fold"]) == 83
        assert int(scope["most_dispersed"]["fold"]) == 144
        assert "glob('*.png')" not in "".join(nb["cells"][9]["source"])
