import sympy
from sympy.parsing.latex import parse_latex

def generate_steps(expr_str):
    expr = parse_latex(expr_str)
    var = list(expr.free_symbols)[0]
    solutions = sympy.solve(expr, var)
    diff = sympy.simplify(expr.lhs - expr.rhs)
    poly = diff.as_poly(var)
    degree = poly.degree() if poly else None

    steps = [f"\\text{{Given equation: }} {sympy.latex(expr)}"]

    if degree is not None and degree >= 3:
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
            for factor, pow_val in factors:
                f_eq = sympy.Eq(factor, 0)
                f_sols = sympy.solve(f_eq, var)
                f_sols_str = ", ".join(sympy.latex(s) for s in f_sols)
                if pow_val > 1:
                    steps.append(f"\\text{{Factor }} ({sympy.latex(factor)})^{{{pow_val}}} = 0 \\implies {sympy.latex(var)} = {f_sols_str}")
                else:
                    steps.append(f"\\text{{Factor }} {sympy.latex(factor)} = 0 \\implies {sympy.latex(var)} = {f_sols_str}")
        except Exception as e:
            print("Factor list err:", e)

        sols_str = ", ".join(sympy.latex(s) for s in solutions) if solutions else "\\text{No solution}"
        steps.append(f"\\text{{Final solutions: }} {sympy.latex(var)} \\in \\{{{sols_str}\\}}")

    for idx, s in enumerate(steps, 1):
        print(f"Step {idx}: {s}")

generate_steps("(x+3)^{4}+(x+1)^{4}=82")
