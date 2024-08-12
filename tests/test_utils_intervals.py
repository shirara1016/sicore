# def test_poly_lt_zero():
#     testcase = [
#         ([1], []),
#         ([0], [[NINF, INF]]),
#         ([-1], [[NINF, INF]]),
#         ([1, 1], [[NINF, -1]]),
#         ([-1, 1], [[1, INF]]),
#         ([1, -2, 2], []),
#         ([1, -2, 1], []),
#         ([1, 0, -1], [[-1, 1]]),
#         ([-1, 0, 1], [[NINF, -1], [1, INF]]),
#         ([-1, 2, -1], [[NINF, INF]]),
#         ([-1, 2, -2], [[NINF, INF]]),
#         ([1, -6, 11, -6], [[NINF, 1], [2, 3]]),
#         ([1, -3, 4, -2], [[NINF, 1]]),
#         ([1, 0, 0, 0], [[NINF, 0]]),
#     ]

#     for coef, expected in testcase:
#         assert_allclose(poly_lt_zero(coef), expected)

#     for coef, expected in testcase:
#         poly = np.poly1d(coef)
#         assert_allclose(poly_lt_zero(poly), expected)
