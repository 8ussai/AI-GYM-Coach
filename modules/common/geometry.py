import math
from typing import Tuple, Optional

Point = Tuple[float, float]

def _safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))

def calculate_angle(a: Point, b: Point, c: Point) -> Optional[float]:
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    dot = bax*bcx + bay*bcy
    mba = math.hypot(bax, bay)
    mbc = math.hypot(bcx, bcy)
    if mba == 0 or mbc == 0:
        return None
    cos_ang = dot / (mba * mbc)
    return math.degrees(_safe_acos(cos_ang))
   
def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def angle_from_vertical(top: Point, bottom: Point) -> Optional[float]:
    vx = top[0] - bottom[0]
    vy = top[1] - bottom[1]
    ang = abs(math.degrees(math.atan2(vx, -vy)))
    return ang