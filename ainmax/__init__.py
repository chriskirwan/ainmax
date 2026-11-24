from ainmax import feidhm as feidhm, nn as nn, random as random, sharding as sharding, typing as typing
from ainmax._src.axis import Axis as Axis
from ainmax._src.axis_util import (
    axis_index as axis_index,
    axis_indices as axis_indices,
    axis_names as axis_names,
    axis_selection_to_tuple as axis_selection_to_tuple,
    axis_shape_to_tuple as axis_shape_to_tuple,
    axis_sizes as axis_sizes,
    axis_spec_to_tuple as axis_spec_to_tuple,
    concatenate_axes as concatenate_axes,
    intersect_axes as intersect_axes,
    is_anonymous_axis as is_anonymous_axis,
    is_axis_compatible as is_axis_compatible,
    is_named_axis as is_named_axis,
    make_axes as make_axes,
    match_axes as match_axes,
    remove_axes as remove_axes,
    replace_axes as replace_axes,
    resolve_axes as resolve_axes,
    union_axes as union_axes,
)
from ainmax._src.config import (
    config as config,
    default_matmul_precision as default_matmul_precision,
    enable_x64 as enable_x64,
    numpy_dtype_promotion as numpy_dtype_promotion,
    short_arrays as short_arrays,
)
from ainmax._src.filters import (
    is_arrayish as is_arrayish,
    is_inexact_arrayish as is_inexact_arrayish,
    is_inexact_jax_array as is_inexact_jax_array,
    is_inexact_jax_array_like as is_inexact_jax_array_like,
    is_inexact_named_array as is_inexact_named_array,
    is_jax_array as is_jax_array,
    is_jax_array_like as is_jax_array_like,
    is_named_array as is_named_array,
    is_scalar as is_scalar,
)
from ainmax._src.hof import fold as fold, scan as scan, vmap as vmap
from ainmax._src.typing import NamedArray as NamedArray, NamedArrayLike as NamedArrayLike
