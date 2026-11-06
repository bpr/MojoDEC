from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import real_vector_from_list


def test_coo_sums_duplicates() raises:
    var A = CSRMatrix.from_coo(
        2, 3, [0, 1, 0, 0], [2, 0, 2, 0], [1.0, 2.0, 3.0, 4.0]
    )
    assert_equal(A.nnz(), 3)
    assert_equal(A[0, 2], 4.0)
    assert_equal(A[0, 0], 4.0)
    assert_equal(A[1, 0], 2.0)
    assert_equal(A[1, 1], 0.0)


def test_transpose_matmul() raises:
    var A = CSRMatrix.from_coo(2, 3, [0, 0, 1], [0, 2, 1], [1.0, 2.0, 3.0])
    var AtA = A.T() @ A
    assert_equal(AtA.nrows, 3)
    assert_equal(AtA[0, 0], 1.0)
    assert_equal(AtA[0, 2], 2.0)
    assert_equal(AtA[2, 2], 4.0)
    assert_equal(AtA[1, 1], 9.0)
    var x = real_vector_from_list([1.0, 1.0, 1.0])
    var y = A @ x
    assert_equal(y.unsafe_get(0), 3.0)
    assert_equal(y.unsafe_get(1), 3.0)


def test_add_select_bmat() raises:
    var I = CSRMatrix.identity(3)
    var S = I + I * 2.0 - I
    assert_equal(S[1, 1], 2.0)
    var sub = S.select([2, 0], [0, 2])
    assert_equal(sub[0, 1], 2.0)
    assert_equal(sub[1, 0], 2.0)
    var blocks = List[List[Optional[CSRMatrix]]]()
    blocks.append([Optional(I.copy()), Optional[CSRMatrix](None)])
    blocks.append([Optional[CSRMatrix](None), Optional(I.copy())])
    var B = CSRMatrix.bmat(blocks)
    assert_equal(B.nrows, 6)
    assert_equal(B[4, 4], 1.0)
    assert_equal(B[4, 1], 0.0)


def test_select_checks_indices() raises:
    var I = CSRMatrix.identity(3)
    with assert_raises(contains="row 3 out of range"):
        _ = I.select([3], [0])
    with assert_raises(contains="row -1 out of range"):
        _ = I.select_rows([-1])
    with assert_raises(contains="column 5 out of range"):
        _ = I.select_cols([0, 5])
    with assert_raises(contains="repeated column"):
        _ = I.select_cols([1, 1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
