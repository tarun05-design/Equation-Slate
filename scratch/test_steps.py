import sympy
from sympy.parsing.latex import parse_latex

def generate_steps(expr, solutions, simplified, numeric_value):
    steps = []
    if isinstance(expr, sympy.Eq):
        free_symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        steps.append(f"Given equation: {sympy.latex(expr)}")
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
                # a*var + b = 0 => a*var = -b => var = -b/a
                steps.append(f"Isolate constant terms: {sympy.latex(a*var)} = {sympy.latex(-b)}")
                if a != 1:
                    steps.append(f"Divide both sides by coefficient ({sympy.latex(a)}): {sympy.latex(var)} = \\frac{{{sympy.latex(-b)}}}{{{sympy.latex(a)}}}")
                final_val = -b / a
                steps.append(f"Final solution: {sympy.latex(var)} = {sympy.latex(final_val)}")
            elif degree == 2:
                a = poly.coeff_monomial(var**2)
                b = poly.coeff_monomial(var)
                c = poly.coeff_monomial(1)
                steps.append(f"Standard form: {sympy.latex(diff)} = 0 \\quad (a={sympy.latex(a)}, b={sympy.latex(b)}, c={sympy.latex(c)})")
                disc = b**2 - 4*a*c
                steps.append(f"Discriminant: \\Delta = b^2 - 4ac = ({sympy.latex(b)})^2 - 4({sympy.latex(a)})({sympy.latex(c)}) = {sympy.latex(disc)}")
                steps.append(f"Apply quadratic formula: {sympy.latex(var)} = \\frac{{-({sympy.latex(b)}) \\pm \\sqrt{{{sympy.latex(disc)}}}}}{{2({sympy.latex(a)})}}")
                sols_str = ", ".join(sympy.latex(s) for s in solutions)
                steps.append(f"Solutions: {sympy.latex(var)} \\in \\{{{sols_str}\\}}")
            else:
                sols_str = ", ".join(sympy.latex(s) for s in solutions)
                steps.append(f"Solutions: {sols_str}")
        else:
            sols_str = ", ".join(sympy.latex(s) for s in solutions)
            steps.append(f"Solutions: {sols_str}")
    else:
        steps.append(f"Given expression: {sympy.latex(expr)}")
        if simplified is not None and str(simplified) != str(expr):
            steps.append(f"Simplified: {sympy.latex(simplified)}")
        if numeric_value is not None:
            steps.append(f"Evaluated numeric value: {numeric_value}")
    return steps

# Test 1: 2x - 3 = -7
e1 = parse_latex("2x - 3 = -7")
s1 = sympy.solve(e1)
print("=== 2x - 3 = -7 ===")
for st in generate_steps(e1, s1, None, None):
    print(" ", st)

# Test 2: x^2 + 3x - 4 = 0
e2 = parse_latex("x^2 + 3x - 4 = 0")
s2 = sympy.solve(e2)
print("\n=== x^2 + 3x - 4 = 0 ===")
for st in generate_steps(e2, s2, None, None):
    print(" ", st)

# Test 3: 2 + 2
e3 = parse_latex("2 + 2")
sim3 = sympy.simplify(e3)
print("\n=== 2 + 2 ===")
for st in generate_steps(e3, [], sim3, "4"):
    print(" ", st)
