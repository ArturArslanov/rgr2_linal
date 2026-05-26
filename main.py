"""
Exact Jordan normal form over algebraic numbers using SymPy.

Allowed library usage:
- symbolic Matrix storage and basic operations;
- characteristic polynomial and polynomial factorization;
- symbolic rank, nullspace, inverse.

Forbidden and NOT used:
- eigenvals/eigenvects;
- SVD;
- jordan_form or any ready Jordan decomposition routine.

Core logic implemented manually:
- block sizes via defects dim Ker((A - lambda I)^k);
- generalized eigenspaces;
- Jordan chain construction;
- transition matrix P such that A * P = P * J.
"""

import random
import sympy as sp


x = sp.Symbol("x")


# =========================================================
# Printing helpers
# =========================================================


def print_title(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# Basic exact linear algebra wrappers
# =========================================================


def as_matrix(A):
    return sp.Matrix(A)



def matrix_rank(A):
    return as_matrix(A).rank()



def defect(A):
    A = as_matrix(A)
    return A.cols - A.rank()



def nullspace(A):
    return as_matrix(A).nullspace()



def is_independent(vectors):
    if not vectors:
        return True
    M = sp.Matrix.hstack(*vectors)
    return M.rank() == len(vectors)



def independent_after_adding(current_basis, new_vectors):
    return is_independent(current_basis + new_vectors)


# =========================================================
# Characteristic roots without eigenvalue routines
# =========================================================


def characteristic_polynomial(A):
    """
    Returns exact characteristic polynomial det(tI - A).
    This uses charpoly, not eigenvalue routines.
    """
    A = as_matrix(A)
    return A.charpoly(x).as_expr()



def algebraic_roots_with_multiplicities(A):
    """
    Finds exact symbolic roots of the characteristic polynomial.

    We factor p(x). For each irreducible factor f(x)^m, roots are represented
    as CRootOf(f, i). If the factor is linear or has radical roots that SymPy
    can express explicitly, roots() may return explicit expressions.

    Returns:
        [(lambda_expr, algebraic_multiplicity), ...]
    """
    p_expr = sp.factor(characteristic_polynomial(A))
    p_poly = sp.Poly(p_expr, x, extension=True)
    factors = sp.factor_list(p_poly.as_expr(), x, extension=True)[1]

    result = []

    for factor_expr, factor_power in factors:
        factor_poly = sp.Poly(factor_expr, x, extension=True)

        explicit_roots = sp.roots(factor_poly.as_expr(), x)

        if explicit_roots and sum(explicit_roots.values()) == factor_poly.degree():
            for root, root_mult_inside_factor in explicit_roots.items():
                result.append((sp.simplify(root), root_mult_inside_factor * factor_power))
        else:
            # Algebraic roots represented exactly as CRootOf.
            # all_roots(radicals=False) returns exact algebraic RootOf/CRootOf objects.
            roots = factor_poly.all_roots(radicals=False)
            for root in roots:
                result.append((root, factor_power))

    # Stable deterministic order by string representation.
    result.sort(key=lambda item: str(item[0]))
    return result


# =========================================================
# Jordan block sizes from defects
# =========================================================


def shifted_matrix(A, lam):
    A = as_matrix(A)
    return A - lam * sp.eye(A.rows)



def jordan_block_sizes(A, lam, algebraic_multiplicity):
    """
    Let N = A - lambda I.
    d_k = dim Ker(N^k).

    Number of Jordan blocks of size >= k:
        b_k = d_k - d_{k-1}

    Number of Jordan blocks of exact size k:
        e_k = b_k - b_{k+1}
    """
    N = shifted_matrix(A, lam)

    d = [0]
    for k in range(1, algebraic_multiplicity + 1):
        d.append(defect(N ** k))

    b = [0] * (algebraic_multiplicity + 2)
    for k in range(1, algebraic_multiplicity + 1):
        b[k] = d[k] - d[k - 1]

    sizes = []
    for k in range(algebraic_multiplicity, 0, -1):
        exact_count = b[k] - b[k + 1]
        for _ in range(exact_count):
            sizes.append(k)

    return sizes


# =========================================================
# Jordan chain construction
# =========================================================


def build_chain_from_top(N, top, size):
    """
    Given top vector w with N^size w = 0,
    construct the ordered chain

        N^(size-1) w, ..., Nw, w.

    With this order, the Jordan block has ones above the diagonal.
    """
    chain = [None] * size
    chain[-1] = top

    for i in range(size - 2, -1, -1):
        chain[i] = sp.simplify(N * chain[i + 1])

    return chain



def is_zero_vector(v):
    return all(sp.simplify(entry) == 0 for entry in v)



def construct_chains_for_eigenvalue(A, lam, block_sizes):
    """
    Builds Jordan chains for a fixed eigenvalue.

    Strategy:
    - process larger blocks first;
    - for block size s, try vectors from Ker(N^s);
    - keep the first generated chain that remains linearly independent
      from all previously chosen chain vectors.
    """
    N = shifted_matrix(A, lam)
    chosen_vectors = []
    chains = []

    for size in block_sizes:
        candidates = nullspace(N ** size)
        added = False

        for top in candidates:
            chain = build_chain_from_top(N, top, size)

            if size > 1 and is_zero_vector(chain[0]):
                continue

            if independent_after_adding(chosen_vectors, chain):
                chains.append(chain)
                chosen_vectors.extend(chain)
                added = True
                break

        if not added:
            raise RuntimeError(
                f"Could not construct a Jordan chain for lambda={lam}, size={size}. "
                "Try a more robust complement-selection routine."
            )

    return chains


# =========================================================
# Main exact Jordan algorithm
# =========================================================


def jordan_form_exact(A):
    """
    Returns exact symbolic matrices J and P such that

        A * P = P * J

    and therefore

        A = P * J * P^{-1}.
    """
    A = as_matrix(A)
    if A.rows != A.cols:
        raise ValueError("Jordan form is defined only for square matrices")

    n = A.rows
    roots = algebraic_roots_with_multiplicities(A)

    J = sp.zeros(n, n)
    P_cols = []
    current_col = 0
    structure = []

    print("Characteristic polynomial:")
    print(sp.factor(characteristic_polynomial(A)))
    print("\nExact roots and algebraic multiplicities:")
    for lam, mult in roots:
        print(f"  lambda = {lam}, mult = {mult}")

    for lam, mult in roots:
        sizes = jordan_block_sizes(A, lam, mult)
        structure.append((lam, sizes))
        chains = construct_chains_for_eigenvalue(A, lam, sizes)

        print(f"\nlambda = {lam}")
        print(f"Jordan block sizes: {sizes}")

        for chain, size in zip(chains, sizes):
            for v in chain:
                P_cols.append(v)

            for i in range(size):
                J[current_col + i, current_col + i] = lam
                if i + 1 < size:
                    J[current_col + i, current_col + i + 1] = 1

            current_col += size

    P = sp.Matrix.hstack(*P_cols)

    if P.det() == 0:
        raise RuntimeError("Constructed P is singular")

    return sp.simplify(J), sp.simplify(P), structure


# =========================================================
# Test generator
# =========================================================


def jordan_block(lam, size):
    B = sp.zeros(size, size)
    for i in range(size):
        B[i, i] = lam
        if i + 1 < size:
            B[i, i + 1] = 1
    return B



def make_jordan_matrix(block_specs):
    """
    block_specs: list of pairs (lambda, size)
    Example:
        [(sp.sqrt(2), 2), (sp.sqrt(2), 1), (3, 1)]
    """
    blocks = [jordan_block(lam, size) for lam, size in block_specs]
    return sp.diag(*blocks)



def random_unimodular_matrix(n, steps=20, entry_bound=3, seed=None):
    """
    Generates an integer unimodular matrix using elementary row operations.
    Determinant is always +1 or -1.
    """
    if seed is not None:
        random.seed(seed)
    if n <= 0:
        raise ValueError("Matrix size must be positive")

    if n == 1:
        return sp.Matrix([[random.choice([1, -1])]])

    P = sp.eye(n)

    for _ in range(steps):
        op = random.choice(["swap", "add", "neg"])

        if op == "swap":
            i, j = random.sample(range(n), 2)
            P.row_swap(i, j)

        elif op == "add":
            i, j = random.sample(range(n), 2)
            k = random.randint(-entry_bound, entry_bound)
            if k == 0:
                k = 1
            P.row_op(i, lambda value, col: value + k * P[j, col])

        elif op == "neg":
            i = random.randrange(n)
            P.row_op(i, lambda value, col: -value)

    return P



def same_jordan_structure(structure_1, structure_2):
    """
    Compares Jordan structures by exact eigenvalues and block sizes.
    """
    def normalize(structure):
        result = []
        for lam, sizes in structure:
            result.append((sp.simplify(lam), tuple(sorted(sizes, reverse=True))))
        result.sort(key=lambda item: str(item[0]))
        return result

    return normalize(structure_1) == normalize(structure_2)



def generated_test(block_specs, seed=0):
    """
    Generate:
        J0 - desired Jordan matrix;
        P0 - random unimodular transition matrix;
        A  = P0 J0 P0^{-1}.

    Then run jordan_form_exact(A) and compare the recovered structure.
    """
    J0 = make_jordan_matrix(block_specs)
    n = J0.rows
    P0 = random_unimodular_matrix(n, seed=seed)
    A = sp.simplify(P0 * J0 * P0.inv())

    print_title("Generated test")
    print("Input block specs:")
    print(block_specs)
    print("\nOriginal J0:")
    sp.pprint(J0)
    print("\nGenerated unimodular P0, det(P0) =", P0.det())
    sp.pprint(P0)
    print("\nGenerated A = P0 * J0 * P0^{-1}:")
    sp.pprint(A)

    J, P, structure = jordan_form_exact(A)

    expected_structure = []
    for lam, size in block_specs:
        found = False
        for idx, (old_lam, sizes) in enumerate(expected_structure):
            if sp.simplify(old_lam - lam) == 0:
                sizes.append(size)
                found = True
                break
        if not found:
            expected_structure.append((lam, [size]))

    print("\nRecovered J:")
    sp.pprint(J)
    print("\nRecovered P:")
    sp.pprint(P)

    print("\nVerification A*P == P*J:", sp.simplify(A * P - P * J) == sp.zeros(n, n))
    print("Recovered structure matches input:", same_jordan_structure(expected_structure, structure))

    return A, J, P, structure


# =========================================================
# Demo cases
# =========================================================


def demo_manual(A, name):
    print_title(name)
    A = as_matrix(A)
    print("A:")
    sp.pprint(A)

    J, P, structure = jordan_form_exact(A)

    print("\nJordan matrix J:")
    sp.pprint(J)

    print("\nTransition matrix P:")
    sp.pprint(P)

    print("\nCheck A*P == P*J:", sp.simplify(A * P - P * J) == sp.zeros(A.rows, A.cols))
    print("det(P) =")
    sp.pprint(sp.factor(P.det()))


if __name__ == "__main__":
    # 1. Matrix with irrational algebraic eigenvalues +-sqrt(2).
    A1 = sp.Matrix([
        [0, 2],
        [1, 0]
    ])

    # 2. Matrix similar to one Jordan block J_2(sqrt(2)).
    lam = sp.sqrt(2)
    J2 = make_jordan_matrix([(lam, 2)])
    P2 = sp.Matrix([
        [1, 2],
        [1, 3]
    ])
    A2 = sp.simplify(P2 * J2 * P2.inv())

    # 3. Mixed symbolic Jordan structure.
    #    Note: A may contain sqrt(2). For integer A with algebraic eigenvalues,
    #    conjugate eigenvalues usually need to appear together.
    J3 = make_jordan_matrix([(sp.sqrt(2), 1), (-sp.sqrt(2), 1), (3, 1)])
    P3 = random_unimodular_matrix(3, seed=10)
    A3 = sp.simplify(P3 * J3 * P3.inv())

    demo_manual(A1, "Exact algebraic eigenvalues: lambda = +-sqrt(2)")
    demo_manual(A2, "One Jordan block J_2(sqrt(2))")
    demo_manual(A3, "Mixed symbolic Jordan structure")

    # Generator tests.
    generated_test([(sp.sqrt(2), 1), (-sp.sqrt(2), 1)], seed=1)
    generated_test([(3, 2), (3, 1), (-1, 1)], seed=2)
    generated_test([(sp.sqrt(3), 2)], seed=3)
