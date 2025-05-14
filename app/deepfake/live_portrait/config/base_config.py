# coding: utf-8

"""
pretty printing class
"""

from __future__ import annotations
import os
from typing import Tuple

liveportrait_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
#repo_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
models_path = os.path.abspath(os.path.join(liveportrait_path, '../../../../models'))

print(f'models_path: {models_path}')
