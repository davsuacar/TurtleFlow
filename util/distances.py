import math

def calculate_distance(x, y):
    math.hypot(y[0] - x[0], y[1] - x[1])

def calculate_proportions(points):
    # Width distances
    point0 = calculate_distance(points[0], points[16])
    point1 = calculate_distance(points[1], points[15])
    point2 = calculate_distance(points[2], points[14])
    point3 = calculate_distance(points[3], points[13])
    point4 = calculate_distance(points[4], points[12])
    point5 = calculate_distance(points[5], points[11])
    point6 = calculate_distance(points[6], points[10])
    point7 = calculate_distance(points[7], points[9])

    # Border distances
    point8 = calculate_distance(points[0], points[1])
    point9 = calculate_distance(points[1], points[2])
    point10 = calculate_distance(points[2], points[3])
    point11 = calculate_distance(points[3], points[4])
    point12 = calculate_distance(points[4], points[5])
    point13 = calculate_distance(points[5], points[6])
    point14 = calculate_distance(points[6], points[7])
    point15 = calculate_distance(points[7], points[8])


    # Cheekbone distances

    point16 = calculate_distance(points[31], points[36])
    point17 = calculate_distance(points[32], points[41])
    point18 = calculate_distance(points[33], points[40])
    point19 = calculate_distance(points[33], points[39])
    point20 = calculate_distance(points[33], points[42])
    point21 = calculate_distance(points[33], points[47])
    point22 = calculate_distance(points[34], points[46])
    point23 = calculate_distance(points[35], points[45])


    # Height distances

    point24 = calculate_distance(points[4], points[17])
    point25 = calculate_distance(points[5], points[18])
    point26 = calculate_distance(points[6], points[19])
    point27 = calculate_distance(points[7], points[20])
    point27 = calculate_distance(points[8], points[21])
    point28 = calculate_distance(points[8], points[22])
    point29 = calculate_distance(points[9], points[23])
    point30 = calculate_distance(points[10], points[24])
    point31 = calculate_distance(points[11], points[25])
    point31 = calculate_distance(points[12], points[26])

    # Nasal Septum distances

    # Nose distances
