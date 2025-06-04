import torch
from src.utils.data_utils import min_max_normalize

def test_min_max_normalize_basic():
    x = torch.rand(3, 3)
    y = min_max_normalize(x)
    assert y.shape == x.shape
    assert torch.isclose(y.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(y.max(), torch.tensor(1.0), atol=1e-6)
