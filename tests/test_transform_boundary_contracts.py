"""Independent axis maps, row-vector matrices and singular rotation limits."""

import math

import numpy as np
import pytest

from geometry import Coord, Point, Quaternion, Q0, Transformation


def _quarter_turn():
    return Quaternion(math.sqrt(.5), 0., 0., math.sqrt(.5))


def test_point_application_rotates_then_translates_with_the_declared_axis_order():
    transform = Transformation(Point(10., -3., 2.), _quarter_turn())
    source = Point([[1., 2., 3.], [-4., 5., -6.]])
    before = source.data.copy()
    # A +90 degree turn about world z maps (x,y,z) to (-y,x,z).
    rotated = [[-2, 1, 3], [-5, -4, -6]]
    expected = [[8, -2, 5], [5, -7, -4]]
    np.testing.assert_allclose(transform.rotate(source).data, rotated, rtol=0, atol=2e-15)
    np.testing.assert_allclose(transform.point(source).data, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(transform.apply(source).data, expected, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(source.data, before)


def test_homogeneous_matrix_acts_on_row_vectors_with_translation_in_the_last_row():
    transform = Transformation(Point(10., -3., 2.), _quarter_turn())
    expected = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [10, -3, 2, 1.]])
    np.testing.assert_allclose(transform.to_matrix()[0], expected, rtol=0, atol=3e-16)
    np.testing.assert_allclose(np.array([1, 2, 3, 1.]) @ transform.to_matrix()[0], [8, -2, 5, 1], rtol=0, atol=1e-15)


def test_point_only_and_attitude_only_constructors_preserve_each_vector_row():
    points = Point([[1., 2., 3.], [4., 5., 6.]])
    translations = Transformation(points)
    np.testing.assert_array_equal(translations.pos.data, points.data)
    np.testing.assert_array_equal(translations.att.data, [[1, 0, 0, 0], [1, 0, 0, 0]])
    rotations = Quaternion([[1., 0., 0., 0.], [0., 0., 0., 1.]])
    attitudes = Transformation(rotations)
    np.testing.assert_array_equal(attitudes.pos.data, np.zeros((2, 3)))
    np.testing.assert_array_equal(attitudes.att.data, rotations.data)
    np.testing.assert_array_equal(Transformation.zero(2).pos.data, np.zeros((2, 3)))


def test_single_translation_broadcasts_over_multiple_attitudes():
    transform = Transformation.build(Point(10., -3., 2.), Quaternion([[1., 0., 0., 0.], [0., 0., 0., 1.]]))
    assert len(transform) == 2
    np.testing.assert_array_equal(transform.pos.data, [[10, -3, 2], [10, -3, 2]])
    np.testing.assert_array_equal(transform.point(Point(1., 2., 3.)).data, [[11, -1, 5], [9, -5, 5]])


def test_coordinate_construction_and_application_match_the_known_axes():
    axes = Coord.from_xy(Point(10., -3., 2.), Point(0., 1., 0.), Point(-1., 0., 0.))
    transform = Transformation.from_coord(axes)
    np.testing.assert_allclose(transform.point(Point(1., 2., 3.)).data, [[8, -2, 5]], rtol=0, atol=2e-15)
    for coord in (transform.coord(), transform.apply(Coord.zero())):
        np.testing.assert_array_equal(coord.origin.data, [[10, -3, 2]])
        np.testing.assert_allclose(coord.x_axis.data, [[0, 1, 0]], rtol=0, atol=3e-16)
        np.testing.assert_allclose(coord.y_axis.data, [[-1, 0, 0]], rtol=0, atol=3e-16)
        np.testing.assert_allclose(coord.z_axis.data, [[0, 0, 1]], rtol=0, atol=3e-16)


def test_transform_application_composes_attitudes_and_offsets_in_world_coordinates():
    transform = Transformation(Point(10., -3., 2.), _quarter_turn())
    other = Transformation(Point(1., 2., 3.), _quarter_turn())
    applied = transform.apply(other)
    np.testing.assert_allclose(applied.pos.data, [[8, -2, 5]], rtol=0, atol=2e-15)
    np.testing.assert_allclose(applied.att.transform_point(Point(1., 2., 3.)).data, [[-1, -2, 3]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(transform.offset(Point(1., 2., 3.)).pos.data, [[11, -1, 5]])
    offset = transform.offset(other)
    np.testing.assert_array_equal(offset.pos.data, [[11, -1, 5]])
    np.testing.assert_allclose(offset.att.transform_point(Point(1., 2., 3.)).data, [[-1, -2, 3]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(transform.apply(Q0()).transform_point(Point(1., 2., 3.)).data, [[-2, 1, 3]], rtol=0, atol=1e-15)
    for attribute in ("x", "y", "z", "rw", "rx", "ry", "rz"):
        assert getattr(transform, attribute).shape == (1,)
    with pytest.raises(AttributeError):
        getattr(transform, "missing")


@pytest.mark.parametrize("method", ["rotate", "offset"])
def test_explicit_transform_operations_reject_unrelated_types(method):
    with pytest.raises(TypeError, match="expected a Point"):
        getattr(Transformation(), method)("invalid")


@pytest.mark.parametrize("sign", [-1, 1])
def test_gimbal_pitch_limit_has_finite_euler_angles_and_the_known_axis_map(sign):
    rotation = Quaternion(math.sqrt(.5), 0., sign * math.sqrt(.5), 0.)
    angles = rotation.to_euler()
    np.testing.assert_allclose(angles.data, [[0, sign * math.pi / 2, 0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(rotation.transform_point(Point(1., 0., 0.)).data, [[0, 0, -sign]], rtol=0, atol=3e-16)
    np.testing.assert_allclose(rotation.transform_point(Point(0., 0., 1.)).data, [[sign, 0, 0]], rtol=0, atol=3e-16)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_half_turn_matrix_conversion_preserves_analytic_axes_without_requiring_a_quaternion_sign(axis):
    diagonal = np.full(3, -1.)
    diagonal[axis] = 1
    rotation = Quaternion.from_rotation_matrix(np.diag(diagonal))
    np.testing.assert_allclose(rotation.transform_point(Point(np.eye(3))).data, np.diag(diagonal), rtol=0, atol=1e-15)


def test_zero_vector_normalization_has_a_finite_zero_limit():
    np.testing.assert_array_equal(Point(0., 0., 0.).unit().data, [[0, 0, 0]])
