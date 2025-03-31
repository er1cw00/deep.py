# coding: utf-8

"""
All configs for user
"""
from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from typing import Optional, Literal
from .printable import Printable

    
import argparse

parser = argparse.ArgumentParser(description="conv.py")
parser.add_argument("--config-file", type=str, default="conv.yaml")

