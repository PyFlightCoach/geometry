"""Intended public contracts: kept diagnostic until their bounded repairs pass."""

import numpy as np
import pytest

from geometry import Point, Q0, Time, Transformation


def test_transform_build_broadcasts_a_single_attitude_over_multiple_positions():
    # geometry/README.md explicitly promises singleton broadcasting for geometric
    # vectors. The converse (one position, many attitudes) already succeeds.
    points = Point([[1., 2., 3.], [4., 5., 6.]])
    transform = Transformation.build(points, Q0())
    np.testing.assert_array_equal(transform.pos.data, points.data)
    np.testing.assert_array_equal(transform.att.data, [[1, 0, 0, 0], [1, 0, 0, 0]])


def test_transform_build_rejects_incompatible_nonsingleton_lengths():
    with pytest.raises(ValueError, match="incompatible lengths"):
        Transformation.build(Point([[1., 2., 3.], [4., 5., 6.]]), Q0(3))


def test_vectorized_transforms_each_produce_their_own_homogeneous_matrix():
    # Base geometric entities support vectors of rows, and the public matrix
    # conversion must preserve those rows instead of collapsing their origins.
    transform = Transformation(Point([[1., 2., 3.], [4., 5., 6.]]))
    matrices = transform.to_matrix()
    assert matrices.shape == (2, 4, 4)
    expected = np.tile(np.eye(4), (2, 1, 1))
    expected[:, 3, :3] = [[1, 2, 3], [4, 5, 6]]
    np.testing.assert_array_equal(matrices, expected)


def test_default_time_interpolator_recalculates_the_sample_intervals():
    # Base.interpolate has an explicit Time default; use the same forward-dt
    # convention required by Time.from_t and the existing dedicated accessor.
    interpolate = Time.from_t([0., 2., 5.]).interpolate()
    np.testing.assert_array_equal(interpolate(np.array([0., .5, 1., 1.5, 2.])).data,
                                  [[0, 1], [1, 1], [2, 1.5], [3.5, 1.5], [5, 0]])
