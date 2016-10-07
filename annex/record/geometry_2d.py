import numpy as np
import math
import itertools
import copy
from external import mask


def is_whole(x):
    if type(x) == float:
        return x.is_integer()
    else:
        if x % 1 == 0:
            return True
        else:
            return False


def is_integral(x):
    if type(x) == int:
        return True
    else:
        return is_whole(x)


def is_iterable_integral(values, strict=False):
    conversion_required = False
    for v in values:
        true_int = isinstance(v, int)
        if not true_int:
            conversion_required = True
            if strict:
                return False, True
            elif not is_whole(v):
                return False, True
    return True, conversion_required


class BoundingBox:

    def __init__(self, xmin=0, ymin=0, w=0, h=0):
        assert w >= 0 and h >= 0
        assert xmin >= 0 and ymin >= 0
        self.xmin = xmin
        self.ymin = ymin
        self.width = w
        self.height = h

    def is_empty(self):
        # no points contained by this bbox
        return self.w <= 0 or self.h <= 0

    def is_integral(self, strict=False):
        return is_iterable_integral(self.as_list(), strict)

    def to_integers(self):
        self.xmin = math.floor(self.xmin)
        self.ymin = math.floor(self.ymin)
        self.width = math.ceil(self.width)
        self.height = math.ceil(self.height)
        return self

    def to_relative(self, scale_width, scale_height):
        self.xmin = float(self.xmin) / scale_width
        self.ymin = float(self.ymin) / scale_height
        self.width = float(self.width) / scale_width
        self.height = float(self.height) / scale_height
        return self

    def to_absolute(self, image_width, image_height, keep_float=False):
        self.xmin = self.xmin * image_width
        self.ymin = self.ymin * image_height
        self.width = self.width * image_width
        self.height = self.height * image_height
        if not keep_float:
            return self.to_integers()
        else:
            return self

    def clip(self, xmin, ymin, w, h):
        self.xmin = max(0, min(self.xmin, xmin))
        self.ymin = max(0, min(self.ymin, ymin))
        self.width = max(0, min(self.width, w))
        self.height = max(0, min(self.height, h))
        return self

    def as_integers(self):
        return copy.copy(self).to_integers()

    def as_list(self, fmt='xywh'):
        if fmt == 'xywh':
            coord_list = [self.xmin, self.ymin, self.width, self.height]
        elif fmt == 'yxhw':
            coord_list = [self.ymin, self.xmin, self.height, self.width]
        elif fmt == 'xyxy':
            coord_list = [self.xmin, self.ymin, self.xmin + self.width - 1, self.ymin + self.height - 1]
        elif fmt == 'yxyx':
            coord_list = [self.ymin, self.xmin, self.ymin + self.height - 1, self.xmin + self.width - 1]
        else:
            assert False, 'Unknown bbox list format'
        return coord_list

    def append_to_lists(self, xmin_list, ymin_list, width_list, height_list):
        xmin_list.append(self.xmin)
        ymin_list.append(self.ymin)
        width_list.append(self.width)
        height_list.append(self.height)

    @classmethod
    def from_points(cls, min_xy, max_xy):
        # Only supporting (x,y) points right now, add other formats and specifier if needed
        return cls(min_xy[0], min_xy[1], max_xy[0] - min_xy[0] + 1, max_xy[1] - min_xy[1] + 1)

    @classmethod
    def from_xyxy(cls, xmin, ymin, xmax, ymax, correct_flipped=False):
        x_flipped = True if xmax >= 0 and xmin > xmax else False
        y_flipped = True if ymax >= 0 and ymin > ymax else False
        if correct_flipped:
            if np.logical_xor(x_flipped, y_flipped):
                assert False, "Invalid bounding box"
            elif x_flipped and y_flipped:
                xmin, xmax = xmax, xmin
                ymin, ymax = ymax, ymin
        return cls(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)

    @classmethod
    def from_list(cls, coord_list, fmt='yxyx'):
        assert len(coord_list) == 4
        if fmt == 'xyxy':
            return cls.from_xyxy(coord_list[0], coord_list[1], coord_list[2], coord_list[3])
        elif fmt == 'yxyx':
            return cls.from_xyxy(coord_list[1], coord_list[0], coord_list[3], coord_list[2])
        elif fmt == 'xywh':
            return cls(coord_list[0], coord_list[1], coord_list[2], coord_list[3])
        elif fmt == 'yxhw':
            return cls(coord_list[1], coord_list[0], coord_list[3], coord_list[2])
        else:
            assert False, 'Unknown bbox list format'


class Polygon2D:
    def __init__(self, points=[]):
        self.points = points

    def is_integral(self, strict=False):
        return is_iterable_integral(itertools.chain(*self.points), strict)

    def to_integers(self):
        self.points = [(math.floor(p[0]), math.ceil(p[1])) for p in self.points]
        return self

    def as_integers(self):
        return copy.copy(self).to_integers()

    def signed_area(self):
        acc = 0
        for i in range(len(self.points)):
            # http://mathworld.wolfram.com/PolygonArea.html
            px1, py1 = self.points[i]
            px2, py2 = self.points[(i + 1) % len(self.points)]
            acc += (px1 * py2) - (px2 * py1)
        return acc / 2

    def is_clockwise(self, y_axis_up=False):
        area = self.signed_area()
        return area < 0 if y_axis_up else area >= 0

    def clip(self, point_min, point_max):
        x_coords, y_coords = self.as_separates()
        x_coords = np.clip(np.asarray(x_coords), point_min[0], point_max[0])
        y_coords = np.clip(np.asarray(y_coords), point_min[1], point_max[1])
        self.points = list(zip(x_coords, y_coords))
        return self

    def as_separates(self):
        return zip(*self.points)

    def append_to_lists(self, lx, ly, delta=False):
        xs, ys = self.as_separates()
        if delta:
            xs = [xs[0]] + [j - i for i, j in zip(xs, xs[1:])]
            ys = [ys[0]] + [j - i for i, j in zip(ys, ys[1:])]
        lx.extend(xs)
        ly.extend(ys)
        return len(xs)

    @classmethod
    def from_list(cls, xylist):
        assert len(xylist) % 2 == 0
        points = list(zip(xylist[0::2], xylist[1::2]))
        return cls(points)

    @classmethod
    def from_separates(cls, x_list, y_list):
        assert len(x_list) == len(y_list)
        points = list(zip(x_list, y_list))
        return cls(points)


class Mask:
    def __init__(self, w, h, mask):
        self.w = w
        self.h = h
        self.mask = mask

    @classmethod
    def from_bytes(cls, list):
        pass


class MaskRle:
    def __init__(self, w, h, rle_bytes=None):
        self.w = w
        self.h = h
        self.rle_bytes = rle_bytes

    @classmethod
    def from_list(cls, w, h, src_list):
        rle = mask.frPyObjects(src_list, h, w)
        assert isinstance(rle, list)
        if len(rle) > 1:
            rle_bytes = mask.merge(rle)['counts']
        else:
            rle_bytes = rle[0]['counts']
        return cls(w, h, rle_bytes)

    @classmethod
    def from_dict(cls, src_dict):
        assert 'counts' in src_dict
        assert 'size' in src_dict
        h = src_dict['size'][0]
        w = src_dict['size'][1]
        rle = mask.frPyObjects([src_dict], h, w)
        assert isinstance(rle, list) and isinstance(rle[0], dict)
        return cls(w, h, rle[0]['counts'])

    def append_to_list(self, dest_list):
        dest_list.append(self.rle_bytes)
