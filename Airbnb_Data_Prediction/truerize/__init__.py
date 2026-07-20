import os
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
import subprocess
import importlib
import contextlib
import io
import warnings
import platform
import json
import html
import argparse
import base64
import csv
import random
import datetime
from pathlib import Path
from typing import Literal

warnings.filterwarnings("ignore")

PY = sys.version_info
OS = platform.system()

DEPENDENCY_GROUPS = [
    (
        "core",
        [
            ("tqdm", "tqdm", False, None),
            ("numpy", "numpy", False, None),
            ("pandas", "pandas", False, None),
            ("scipy", "scipy", False, None),
            ("joblib", "joblib", False, None),
            ("matplotlib", "matplotlib", False, None),
            ("seaborn", "seaborn", False, None),
            ("plotly", "plotly", False, None),
        ],
    ),
    (
        "data",
        [
            ("polars", "polars", False, (3, 8)),
            ("pyarrow", "pyarrow", False, None),
            ("duckdb", "duckdb", False, None),
        ],
    ),
    (
        "ml",
        [
            ("sklearn", "scikit-learn", False, None),
            ("xgboost", "xgboost", False, None),
            ("lightgbm", "lightgbm", False, None),
        ],
    ),
    (
        "xai",
        [
            ("shap", "shap", False, None),
            ("lime", "lime", False, None),
            ("pyod", "pyod", False, None),
        ],
    ),
    (
        "validation",
        [
            ("great_expectations", "great_expectations", True, (3, 8)),
            ("cerberus", "cerberus", False, None),
            ("pydantic", "pydantic", False, None),
        ],
    ),
    (
        "ui_and_docs",
        [
            ("docx", "python-docx", False, None),
            ("html2image", "html2image", True, None),
            ("IPython", "ipython", False, None),
            ("ipywidgets", "ipywidgets", False, None),
            ("panel", "panel", False, (3, 8)),
        ],
    ),
]

IMPORT_FAILURES = {}
INSTALL_FAILURES = {}


class _UnavailableDependency:
    def __init__(self, module_name, error):
        self._module_name = module_name
        self._error = error

    def __getattr__(self, name):
        raise ImportError(
            f"Optional dependency '{self._module_name}' is unavailable in this environment. "
            f"Original error: {self._error}"
        ) from self._error

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"Optional dependency '{self._module_name}' is unavailable in this environment. "
            f"Original error: {self._error}"
        ) from self._error


def _bootstrap_pip():
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        return
    except Exception:
        pass
    try:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)
        return
    except Exception:
        pass
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
    except Exception:
        pass


def _try_import(module_name):
    try:
        importlib.invalidate_caches()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return importlib.import_module(module_name)
    except Exception as exc:
        IMPORT_FAILURES[module_name] = exc
        return None


def _install_package(pip_name):
    name_map = {
        "scikit-learn": "sklearn",
        "python-docx": "docx",
        "html2image": "html2image",
    }
    module_name = name_map.get(pip_name, pip_name.replace("-", "_"))
    commands = [
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet", "--no-warn-script-location"],
        [sys.executable, "-m", "pip", "install", pip_name, "--user", "--quiet", "--no-warn-script-location"],
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet", "--ignore-requires-python"],
        [sys.executable, "-m", "pip", "install", pip_name, "--force-reinstall", "--quiet", "--no-deps"],
    ]

    for command in commands:
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=600,
                check=False,
            )
            if _try_import(module_name) is not None:
                return True
        except Exception:
            continue
    return False


def _install_dependency_groups():
    _bootstrap_pip()
    for _, dependencies in DEPENDENCY_GROUPS:
        for module_name, pip_name, optional, min_py in dependencies:
            if min_py and PY < min_py:
                continue

            imported = _try_import(module_name)
            if imported is not None:
                IMPORT_FAILURES.pop(module_name, None)
                continue

            installed = _install_package(pip_name)
            imported = _try_import(module_name)

            if imported is not None:
                IMPORT_FAILURES.pop(module_name, None)
                continue

            error = IMPORT_FAILURES.get(module_name)
            INSTALL_FAILURES[module_name] = error
            if not optional:
                raise ImportError(
                    f"[truerize] Required dependency '{pip_name}' could not be loaded."
                ) from error


with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    _install_dependency_groups()

import tqdm
import tqdm.std
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="IProgress not found.*")
    import tqdm.auto


class _SilentTqdm:
    @staticmethod
    def format_interval(seconds):
        return "00:00"

    @staticmethod
    def format_meter(*args, **kwargs):
        return ""

    @staticmethod
    def format_sizeof(num, suffix="", divisor=1000):
        try:
            return str(round(float(num), 2))
        except Exception:
            return "0"

    @staticmethod
    def status_printer(*args, **kwargs):
        return lambda *a, **kw: None

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.total = kwargs.get("total", 0)
        self.n = 0
        self.disable = True
        self.desc = kwargs.get("desc", "")

    def __iter__(self):
        return iter(self.iterable or [])

    def update(self, n=0):
        self.n += n or 0

    def close(self):
        return None

    def refresh(self, *args, **kwargs):
        return None

    def set_description(self, *args, **kwargs):
        return None

    def set_postfix(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        self.n = 0

    def display(self, *args, **kwargs):
        return None

    def clear(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _silent_trange(*args, **kwargs):
    try:
        iterable = range(*args)
    except Exception:
        iterable = []
    return _SilentTqdm(iterable=iterable, **kwargs)


tqdm.tqdm = _SilentTqdm
tqdm.std.tqdm = _SilentTqdm
tqdm.trange = _silent_trange
tqdm.auto.tqdm = _SilentTqdm
tqdm.auto.trange = _silent_trange

try:
    import tqdm.autonotebook

    tqdm.autonotebook.tqdm = _SilentTqdm
    tqdm.autonotebook.trange = _silent_trange
except Exception:
    pass

try:
    import tqdm.notebook

    tqdm.notebook.tqdm = _SilentTqdm
    tqdm.notebook.tqdm_notebook = _SilentTqdm
    tqdm.notebook.trange = _silent_trange
except Exception:
    pass

for module_name in ("tqdm", "tqdm.std", "tqdm.auto", "tqdm.autonotebook", "tqdm.notebook"):
    module = sys.modules.get(module_name)
    if module is not None:
        try:
            module.tqdm = _SilentTqdm
        except Exception:
            pass
        try:
            module.trange = _silent_trange
        except Exception:
            pass

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import gaussian_kde

import polars as pl
import polars.selectors as cs
import pyarrow as pa
import duckdb

from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

import shap
import lime
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import pyod
from pyod.models.iforest import IForest

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as plotly_express
import plotly.graph_objects as plotly_graph_objects
px = plotly_express
go = plotly_graph_objects
sp = scipy

import joblib
import panel
import IPython
import ipywidgets as widgets

try:
    from docx import Document
    from docx.shared import Inches, Pt
except Exception as exc:
    raise ImportError("[truerize] python-docx loaded but submodules could not be imported.") from exc

try:
    import html2image as _html2image_module

    Html2Image = getattr(_html2image_module, "Html2Image", None)
except Exception as exc:
    IMPORT_FAILURES["html2image"] = exc
    _html2image_module = _UnavailableDependency("html2image", exc)
    Html2Image = _UnavailableDependency("html2image.Html2Image", exc)

try:
    from cerberus import Validator
except Exception as exc:
    raise ImportError("[truerize] cerberus loaded but Validator could not be imported.") from exc

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        StrictFloat,
        StrictInt,
        StrictStr,
        ValidationError,
        create_model,
        field_validator,
    )
except Exception:
    from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr, ValidationError, create_model, validator as field_validator

    class ConfigDict(dict):
        pass

from IPython import get_ipython as _get_ipython


def get_ipython():
    return _get_ipython()


try:
    import great_expectations as gx
    from great_expectations.core.batch import Batch
    from great_expectations.execution_engine import PandasExecutionEngine
    from great_expectations.render.renderer import ExpectationSuitePageRenderer, ValidationResultsPageRenderer
    from great_expectations.render.view import DefaultJinjaPageView
    from great_expectations.validator.validator import Validator as GXValidator

    try:
        import great_expectations.validator.validation_graph as _gx_validation_graph

        _gx_validation_graph.tqdm = _SilentTqdm
    except Exception:
        pass

    try:
        import great_expectations.validator.metrics_calculator as _gx_metrics_calculator

        _gx_metrics_calculator.tqdm = _SilentTqdm
    except Exception:
        pass

    try:
        import great_expectations.validator.validator as _gx_validator_module

        _gx_validator_module.tqdm = _SilentTqdm
    except Exception:
        pass
except Exception as exc:
    IMPORT_FAILURES["great_expectations"] = exc
    gx = _UnavailableDependency("great_expectations", exc)
    DefaultJinjaPageView = _UnavailableDependency("great_expectations.render.view", exc)
    ValidationResultsPageRenderer = _UnavailableDependency("great_expectations.render.renderer", exc)
    ExpectationSuitePageRenderer = _UnavailableDependency("great_expectations.render.renderer", exc)
    Batch = _UnavailableDependency("great_expectations.core.batch", exc)
    PandasExecutionEngine = _UnavailableDependency("great_expectations.execution_engine", exc)
    GXValidator = _UnavailableDependency("great_expectations.validator.validator", exc)


def get_cwd():
    return Path.cwd().resolve()


def make_project_folders(folders=("data", "images", "reports", "outputs", "models", "logs")):
    root = Path.cwd()
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)


def dependency_report():
    items = {
        "numpy": np,
        "pandas": pd,
        "scipy": sp,
        "matplotlib": matplotlib,
        "seaborn": sns,
        "plotly": plotly_express,
        "polars": pl,
        "pyarrow": pa,
        "duckdb": duckdb,
        "scikit-learn": sys.modules.get("sklearn"),
        "xgboost": xgb,
        "lightgbm": lgb,
        "shap": shap,
        "lime": lime,
        "pyod": pyod,
        "great_expectations": None if isinstance(gx, _UnavailableDependency) else gx,
        "cerberus": Validator,
        "pydantic": BaseModel,
        "python-docx": Document,
        "html2image": None if isinstance(Html2Image, _UnavailableDependency) else Html2Image,
        "IPython": IPython,
        "ipywidgets": widgets,
        "panel": panel,
        "joblib": joblib,
    }
    ok = sum(value is not None for value in items.values())
    fail = len(items) - ok
    print(f"\n  Truerize | Python {'.'.join(str(x) for x in PY[:3])} | {OS}")
    print(f"  {ok} loaded  |  {fail} missing\n")
    for name, module in items.items():
        status = "OK     " if module is not None else "MISSING"
        print(f"  {status}  {name}")
    print()


def reinstall_failed():
    for _, dependencies in DEPENDENCY_GROUPS:
        for module_name, pip_name, optional, min_py in dependencies:
            if min_py and PY < min_py:
                continue
            if _try_import(module_name) is None:
                installed = _install_package(pip_name)
                if not installed and not optional:
                    raise ImportError(f"[truerize] Unable to reinstall required dependency: {pip_name}")


__all__ = [
    "os",
    "sys",
    "subprocess",
    "importlib",
    "contextlib",
    "io",
    "warnings",
    "platform",
    "json",
    "html",
    "argparse",
    "base64",
    "csv",
    "Path",
    "Literal",
    "random",
    "datetime",
    "np",
    "pd",
    "sp",
    "pl",
    "cs",
    "pa",
    "duckdb",
    "stats",
    "gaussian_kde",
    "scipy",
    "metrics",
    "preprocessing",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "LinearRegression",
    "LogisticRegression",
    "train_test_split",
    "cross_val_score",
    "GridSearchCV",
    "StratifiedKFold",
    "Pipeline",
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "accuracy_score",
    "mean_squared_error",
    "classification_report",
    "confusion_matrix",
    "compute_class_weight",
    "SVC",
    "xgb",
    "XGBClassifier",
    "lgb",
    "LGBMRegressor",
    "LGBMClassifier",
    "shap",
    "lime",
    "lime_tabular",
    "LimeTabularExplainer",
    "pyod",
    "IForest",
    "matplotlib",
    "plt",
    "sns",
    "plotly_express",
    "plotly_graph_objects",
    "px",
    "go",
    "panel",
    "gx",
    "DefaultJinjaPageView",
    "ValidationResultsPageRenderer",
    "ExpectationSuitePageRenderer",
    "Batch",
    "PandasExecutionEngine",
    "GXValidator",
    "Validator",
    "Document",
    "Inches",
    "Pt",
    "Html2Image",
    "joblib",
    "BaseModel",
    "ConfigDict",
    "StrictFloat",
    "StrictInt",
    "StrictStr",
    "create_model",
    "field_validator",
    "ValidationError",
    "tqdm",
    "widgets",
    "IPython",
    "get_ipython",
    "get_cwd",
    "make_project_folders",
    "dependency_report",
    "reinstall_failed",
]

print("truerize package and __init__.py file created successfully and imported.")

if IMPORT_FAILURES:
    failed_modules = ", ".join(sorted(IMPORT_FAILURES))
    warnings.warn(
        "truerize imported with limited optional dependency support. "
        f"Unavailable or incompatible modules: {failed_modules}.",
        RuntimeWarning,
        stacklevel=2,
    )
