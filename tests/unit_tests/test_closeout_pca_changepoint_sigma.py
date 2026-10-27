"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_changepoint_sigma_reaches_smoothing_std(monkeypatch):
    import moseq2_pca.util as util
    seen = []
    original = util.gauss_smooth
    def record(signal, *args, **kwargs):
        seen.append(kwargs.get("sig", 1.5))
        return original(signal, *args, **kwargs)
    monkeypatch.setattr(util, "gauss_smooth", record)
    util.get_changepoints(np.sin(np.arange(120)[None, :] / 6), sigma=2.)
    assert seen == [2.]
