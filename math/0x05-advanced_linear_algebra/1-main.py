#!/usr/bin/env python3
"""
- [Got]
None
None

(10 chars long)
[stderr]: 
(0 chars long)
[Expected]
[
    [7131, 26840, -33301, 31889, 28240],
    [49662, 7848, 1494, 153162, 1800],
    [4371, 8032, -6437, 38209, -16456],
    [31647, 20768, -3697, 34061, 11704],
    [-73743, -22360, 42041, -72229, -15224]
]
[
    [60160, -15874, 34532, 63080, 27686],
    [22970, -4526, 15412, 21316, 9658],
    [-62518, 18304, -37286, -64790, -27164],
    [-40574, 8846, -20308, -37768, -15436],
    [48148, -12556, 30101, 53294, 19979]
]

(381 chars long)
[stderr]: [Anything]
(0 chars long) [Diff had to be removed because it was too long]
"""

if __name__ == '__main__':
    minor = __import__('1-minor').minor

    mat = [[10, 4, 7, 3, -9],
           [-2, 8, 3, -5, 6],
           [5, 19, 6, 1, 25],
           [7, -30, 21, 4, -1],
           [8, 9, -10, 2, -3]]
    print(minor(mat))

    mat = [[5, 11, 6, 3, -20],
           [1, -9, 13, 8, 5],
           [2, 22, 4, 7, -6],
           [-10, 3, 7, -1, 9],
           [4, 8, -2, 10, 12]]

    print(minor(mat))
