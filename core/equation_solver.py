"""
core/equation_solver.py
------------------------
Converts a recognized LaTeX string into a solved/simplified result using
SymPy.

Flow:
    1. Parse the LaTeX string into a SymPy expression (`parse_latex`, which
       relies on `antlr4-python3-runtime`).
    2. If the expression contains a top-level `=`, treat it as an equation
       and solve for its free symbol(s). If it has no `=`, treat it as a
       bare expression and simplify/evaluate it numerically when possible.
    3. Return a structured, JSON-serializable result, or raise a typed
       `SolverError` with a message that's safe (and useful) to show a user
       -- e.g. "this expression is malformed" rather than a raw traceback.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import sympy
from sympy.parsing.latex import parse_latex

from utils.logger import get_logger

logger = get_logger(__name__)


class SolverError(Exception):
    """Raised when a LaTeX string cannot be parsed or solved."""


@dataclass
class SolveResult:
    original_latex: str
    sympy_input: str
    kind: str                      # "equation" | "expression"
    solutions: List[str] = field(default_factory=list)
    simplified: Optional[str] = None
    numeric_value: Optional[str] = None
    steps: List[str] = field(default_factory=list)


import re


def _clean_latex_for_parsing(latex: str) -> str:
    """Light normalization to improve parse_latex's success rate."""
    cleaned = latex.strip()
    cleaned = cleaned.strip("$").strip()

    # Strip environment wrappers that trip up SymPy parse_latex
    cleaned = re.sub(r"\\begin\{(?:array|matrix|pmatrix|bmatrix)\}(?:\{[^}]*\})?", "", cleaned)
    cleaned = re.sub(r"\\end\{(?:array|matrix|pmatrix|bmatrix)\}", "", cleaned)

    # Strip font styling macros like \mathit{...}, \mathrm{...}, \mathbf{...}, \mathcal{...}, \text{...}
    cleaned = re.sub(r"\\(?:math(?:it|rm|bf|cal|tt|sf|bb)|text|operatorname)\{([^}]*)\}", r"\1", cleaned)
    # Strip unbracketed old-style TeX font switches like \bf, \it, \rm, \sf, \tt, \cal, \bb
    cleaned = re.sub(r"\\(?:bf|it|rm|sf|tt|cal|bb)(?=[^a-zA-Z]|$)\s*", "", cleaned)

    # Strip accent & arrow wrappers like \stackrel{...}{...}, \overline{...}, \vec{...}, \hat{...}
    cleaned = re.sub(r"\\stackrel\{[^}]*\}\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\(?:overline|vec|hat|bar|tilde)\{([^}]*)\}", r"\1", cleaned)

    replacements = {
        r"\left(": "(",
        r"\right)": ")",
        r"\left[": "[",
        r"\right]": "]",
        r"\cdot": "*",
        r"\times": "*",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.strip()


def _generate_solution_steps(
    expr: sympy.Basic,
    solutions: list,
    simplified: Optional[sympy.Basic] = None,
    numeric_value: Optional[str] = None,
) -> List[str]:
    """Generate clear, step-by-step mathematical derivations formatted in LaTeX."""
    steps: List[str] = []
    if isinstance(expr, sympy.Eq):
        free_symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        steps.append(f"\\text{{Given equation: }} {sympy.latex(expr)}")
        if len(free_symbols) == 1:
            var = free_symbols[0]
            diff = sympy.simplify(expr.lhs - expr.rhs)
            try:
                poly = diff.as_poly(var)
                degree = poly.degree() if poly else None
            except Exception:
                degree = None

            if degree == 1:
                a = poly.coeff_monomial(var)
                b = diff.subs(var, 0)
                steps.append(f"\\text{{Isolate constant terms: }} {sympy.latex(a*var)} = {sympy.latex(-b)}")
                if a != 1:
                    steps.append(
                        f"\\text{{Divide by coefficient }} ({sympy.latex(a)}): {sympy.latex(var)} = \\frac{{{sympy.latex(-b)}}}{{{sympy.latex(a)}}}"
                    )
                final_val = -b / a
                steps.append(f"\\text{{Final solution: }} {sympy.latex(var)} = {sympy.latex(final_val)}")
            elif degree == 2:
                a = poly.coeff_monomial(var**2)
                b = poly.coeff_monomial(var)
                c = poly.coeff_monomial(1)
                steps.append(
                    f"\\text{{Standard form: }} {sympy.latex(diff)} = 0 \\quad (a={sympy.latex(a)}, b={sympy.latex(b)}, c={sympy.latex(c)})"
                )
                disc = b**2 - 4 * a * c
                steps.append(
                    f"\\text{{Discriminant: }} \\Delta = b^2 - 4ac = ({sympy.latex(b)})^2 - 4({sympy.latex(a)})({sympy.latex(c)}) = {sympy.latex(disc)}"
                )
                steps.append(
                    f"\\text{{Apply quadratic formula: }} {sympy.latex(var)} = \\frac{{-({sympy.latex(b)}) \\pm \\sqrt{{{sympy.latex(disc)}}}}}{{2({sympy.latex(a)})}}"
                )
                sols_str = ", ".join(sympy.latex(s) for s in solutions)
                steps.append(f"\\text{{Solutions: }} {sympy.latex(var)} \\in \\{{{sols_str}\\}}")
            elif degree is not None and degree >= 3:
                expanded_lhs = sympy.expand(expr.lhs)
                if expanded_lhs != expr.lhs:
                    steps.append(f"\\text{{Expand left-hand side: }} {sympy.latex(expanded_lhs)} = {sympy.latex(expr.rhs)}")

                expanded_diff = sympy.expand(diff)
                if str(expanded_diff) != str(expanded_lhs) or expr.rhs != 0:
                    steps.append(f"\\text{{Standard form: }} {sympy.latex(expanded_diff)} = 0")

                factored_diff = sympy.factor(expanded_diff)
                if str(factored_diff) != str(expanded_diff):
                    steps.append(f"\\text{{Factor polynomial: }} {sympy.latex(factored_diff)} = 0")

                try:
                    coeff, factors = sympy.factor_list(expanded_diff)
                    if len(factors) > 1 or (len(factors) == 1 and factors[0][1] > 1):
                        for factor, pow_val in factors:
                            f_eq = sympy.Eq(factor, 0)
                            f_sols = sympy.solve(f_eq, var)
                            f_sols_str = ", ".join(sympy.latex(s) for s in f_sols)
                            if pow_val > 1:
                                steps.append(f"\\text{{Factor }} ({sympy.latex(factor)})^{{{pow_val}}} = 0 \\implies {sympy.latex(var)} = {f_sols_str}")
                            else:
                                steps.append(f"\\text{{Factor }} {sympy.latex(factor)} = 0 \\implies {sympy.latex(var)} = {f_sols_str}")
                except Exception:
                    pass

                sols_str = ", ".join(sympy.latex(s) for s in solutions) if solutions else "\\text{No solution}"
                steps.append(f"\\text{{Final solutions: }} {sympy.latex(var)} \\in \\{{{sols_str}\\}}")
            else:
                sols_str = ", ".join(sympy.latex(s) for s in solutions) if solutions else "\\text{No solution}"
                steps.append(f"\\text{{Solutions: }} {sols_str}")
        else:
            sols_str = ", ".join(sympy.latex(s) for s in solutions) if solutions else "\\text{No solution}"
            steps.append(f"\\text{{Solutions: }} {sols_str}")
    else:
        steps.append(f"\\text{{Given expression: }} {sympy.latex(expr)}")
        if simplified is not None and str(simplified) != str(expr):
            steps.append(f"\\text{{Simplified: }} {sympy.latex(simplified)}")
        if numeric_value is not None:
            steps.append(f"\\text{{Evaluated numeric value: }} {numeric_value}")
    return steps


def parse_latex_expression(latex: str) -> sympy.Basic:
    """Parse a LaTeX string into a SymPy expression or Eq object."""
    if not latex or not latex.strip():
        raise SolverError("No LaTeX expression to parse.")

    cleaned = _clean_latex_for_parsing(latex)

    try:
        expr = parse_latex(cleaned)
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse_latex failed for %r: %s", latex, exc)
        raise SolverError(
            "Could not parse the recognized expression as valid mathematics. "
            "The handwriting may have been misread -- try a clearer image."
        ) from exc

    return expr


def solve_expression(latex: str) -> SolveResult:
    """Parse and solve/simplify a recognized LaTeX equation or expression."""
    expr = parse_latex_expression(latex)
    cleaned = _clean_latex_for_parsing(latex)

    # `parse_latex` turns "a = b" into an Eq(a, b) automatically when the
    # source contains a top-level "=".
    if isinstance(expr, sympy.Eq):
        free_symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        try:
            solutions = sympy.solve(expr, *free_symbols) if free_symbols else []
        except NotImplementedError as exc:
            raise SolverError(
                "SymPy could not solve this equation symbolically."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("sympy.solve failed for %r: %s", latex, exc)
            raise SolverError("This equation could not be solved.") from exc

        readable_solutions = [str(sol) for sol in solutions] if solutions else []
        steps = _generate_solution_steps(expr, solutions)
        return SolveResult(
            original_latex=latex,
            sympy_input=cleaned,
            kind="equation",
            solutions=readable_solutions,
            simplified=str(sympy.simplify(expr.lhs - expr.rhs)),
            steps=steps,
        )

    # Bare expression: simplify symbolically and, if it has no free
    # variables, also compute a numeric value.
    try:
        simplified = sympy.simplify(expr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sympy.simplify failed for %r: %s", latex, exc)
        raise SolverError("This expression could not be simplified.") from exc

    numeric_value = None
    if not simplified.free_symbols:
        try:
            numeric_value = str(sympy.nsimplify(simplified).evalf(10))
        except Exception:  # noqa: BLE001
            numeric_value = None

    steps = _generate_solution_steps(expr, [], simplified=simplified, numeric_value=numeric_value)

    return SolveResult(
        original_latex=latex,
        sympy_input=cleaned,
        kind="expression",
        simplified=str(simplified),
        numeric_value=numeric_value,
        steps=steps,
    )
