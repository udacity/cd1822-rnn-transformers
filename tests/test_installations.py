import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import scipy


class TestPyTorchInstallation:
    def test_pytorch_version(self):
        assert torch.__version__ == "1.13.1", "PyTorch version is not 1.13.1"

    def test_cuda_available(self):
        assert torch.cuda.is_available(), "CUDA is not available"

    def test_tensor_addition(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        assert z.tolist() == [5, 7, 9], "Tensor addition is incorrect"


class TestLibrariesInstallation:
    def test_numpy_version(self):
        assert np.__version__ == "1.23.5", "NumPy version is not 1.21.3"

    def test_pandas_version(self):
        assert pd.__version__ == "1.5.3", "Pandas version is not 1.3.5"

    def test_matplotlib_version(self):
        assert matplotlib.__version__ == "3.7.0", "Matplotlib version is not 3.5.1"

    def test_sklearn_version(self):
        assert sklearn.__version__ == "1.2.1", "Scikit-learn version is not 1.0.1"

    def test_scipy_version(self):
        assert scipy.__version__ == "1.10.0", "SciPy version is not 1.7.3"

