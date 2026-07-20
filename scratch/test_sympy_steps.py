import sympy

x = sympy.Symbol('x')
eq = sympy.Eq((x+3)**4 + (x+1)**4, 82)
lhs = eq.lhs
rhs = eq.rhs

print("LHS:", lhs)
expanded_lhs = sympy.expand(lhs)
print("Expanded LHS:", expanded_lhs)
diff = sympy.simplify(lhs - rhs)
print("Diff:", diff)
expanded_diff = sympy.expand(diff)
print("Expanded Diff:", expanded_diff)
factored_diff = sympy.factor(expanded_diff)
print("Factored Diff:", factored_diff)

sols = sympy.solve(eq, x)
print("Solutions:", sols)

# Check factors
factors = sympy.factor_list(expanded_diff)
print("Factor list:", factors)
