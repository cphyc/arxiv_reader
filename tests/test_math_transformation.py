import os

import pytest

from arxiv_reader.latex_utils import latex2speech

LATEX_REF_FILE = os.path.join("tests", "maths.tex")
TXT_REF_FILE = os.path.join("tests", "maths.txt")


@pytest.fixture
def latex_content():
    with open(LATEX_REF_FILE) as f:
        return f.read()


@pytest.fixture
def txt_content():
    with open(TXT_REF_FILE) as f:
        return f.read()


def test_math_conversion(latex_content, txt_content):
    content = latex2speech(latex_content)
    for expected, actual in zip(txt_content.split("\n"), content.split("\n")):
        assert expected == actual
