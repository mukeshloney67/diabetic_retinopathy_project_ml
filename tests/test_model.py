from __future__ import annotations

import pytest

from dr_classifier.config import load_config
from dr_classifier.model import build_model, configure_tensorflow


def test_inception_model_has_five_severity_outputs(tmp_path, make_test_config):
    pytest.importorskip("tensorflow")
    config = load_config(make_test_config(tmp_path))
    configure_tensorflow(config)

    model = build_model(config)

    assert model.input_shape == (None, 299, 299, 3)
    assert model.output_shape == (None, 5)
