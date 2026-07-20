from core.equation_solver import solve_expression, SolverError

def test_differential_form():
    latex = r"\left(x^{4}+2y\right)\,d x-x d y=0"
    res = solve_expression(latex)
    assert res.kind == "equation"
    assert len(res.solutions) > 0
    assert "C1" in res.solutions[0] or "C_{1}" in str(res.steps)
    assert any("Classification" in s for s in res.steps)

def test_fractional_derivative():
    latex = r"\frac{dy}{dx} - \frac{2}{x}y = x^3"
    res = solve_expression(latex)
    assert res.kind == "equation"
    assert len(res.solutions) > 0
    assert "C1" in res.solutions[0] or "x**2" in res.solutions[0]

def test_prime_notation_first_order():
    latex = r"y' + 2y = 0"
    res = solve_expression(latex)
    assert res.kind == "equation"
    assert len(res.solutions) > 0
    assert "exp(-2*x)" in res.solutions[0] or "exp" in res.solutions[0]

def test_prime_notation_second_order():
    latex = r"y'' + 4y = 0"
    res = solve_expression(latex)
    assert res.kind == "equation"
    assert len(res.solutions) > 0
    assert "sin" in res.solutions[0] or "cos" in res.solutions[0]

def test_standard_algebraic_equation():
    latex = r"x^2 - 4 = 0"
    res = solve_expression(latex)
    assert res.kind == "equation"
    assert set(res.solutions) == {"-2", "2"}
