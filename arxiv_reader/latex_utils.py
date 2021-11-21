import re

from plasTeX.Base.LaTeX import Math
from plasTeX.Renderers.Text import TextRenderer
from plasTeX.TeX import TeX

Math.odot.str = "sun"
Math.sim.str = " approximately "
Math.ell.str = "l"
Math.star.str = "star"

NUMBER_RE = re.compile(r"^[\+-]?\d+(\.\d*)?$")


class MyRenderer(TextRenderer):
    do_mathrm = do_mathcal = do_mathsc = TextRenderer.default

    def do_lesssim(self, node):
        return "≤"

    def do_gtrsim(self, node):
        return "≥"

    def do_leqslant(self, node):
        return "≤"

    def do_geqslant(self, node):
        return "≥"

    def do_log(self, node):
        return "log"

    def do_in(self, node):
        return " in "

    def do_dot(self, node):
        return "dot"

    def do_bar(self, node):
        return "bar"

    def do__superscript(self, node):
        content = super().do__superscript(node)

        if NUMBER_RE.match(content):
            return "^" + content
        else:
            return content

    def do_math(self, node):
        return str(node)


MATH_RE = re.compile(r"\$(?P<content>([^\$]|(?<=\\)\$)*)\$")


def clean_math(match) -> str:
    txt = match.group("content")

    UNITS_SPELLED_BASE = {
        "pc": "parsec",
        "ly": "light year",
        "m": "meter",
        "y": "year",
        "G": "Gauss",
    }
    UNITS_SPELLED = {}
    for unit, unit_spelled in UNITS_SPELLED_BASE.items():
        for prefix, prefix_spelled in {
            "m": "milli",
            "c": "centi",
            "k": "kilo",
            "M": "mega",
            "G": "giga",
            "μ": "μ",
            "": "",
        }.items():
            UNITS_SPELLED[f"{prefix}{unit}"] = f"{prefix_spelled} {unit_spelled}"
    UNITS_SPELLED["au"] = "astronomical unit"

    for unit, unit_spelled in UNITS_SPELLED.items():
        txt = re.sub("\b%s\b" % unit, unit_spelled, txt)
    return txt


def clean_str(text: str) -> str:
    """Clean a text from any math strings."""
    # Find all math expressions and replace them
    math_cleaned = MATH_RE.subn(clean_math, text)[0]
    return math_cleaned.replace(r"\!", " ").replace(r"\;", " ").replace(r"\,", " ")


def latex2speech(src: str) -> str:
    tex = TeX()
    tex.ownerDocument.config["files"]["split-level"] = -100
    tex.ownerDocument.config["files"]["filename"] = "test.xml"

    # First strip spacings
    txt = clean_str(src)

    tex.input(
        r"""
    \documentclass{article}
    \usepackage{amsmath}
    \usepackage{amssymb}
    \begin{document}
    %s
    \end{document}
    """
        % txt
    )
    document = tex.parse()

    # Render the document
    renderer = MyRenderer()
    renderer.render(document)

    for fname in renderer.files.values():
        if fname == "test.xml":
            with open(fname) as fd:
                txt = fd.read().replace("_", "")
                break
    else:
        raise FileNotFoundError("Could not find file 'test.xml'")

    return txt


if __name__ == "__main__":
    import sys

    sys.argv.pop(0)
    with open(sys.argv[0]) as fd:
        src = fd.read()

    print(latex2speech(src))
