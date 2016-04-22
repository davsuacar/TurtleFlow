import math

def calculate_distance(x, y):
    return math.hypot(y[0] - x[0], y[1] - x[1])

def calculate_proportions(points):

    distances = [0 for i in range(0, 69)]

    # Width distances
    distances[0] = calculate_distance(points[0], points[16])
    distances[1] = calculate_distance(points[1], points[15])
    distances[2] = calculate_distance(points[2], points[14])
    distances[3] = calculate_distance(points[3], points[13])
    distances[4] = calculate_distance(points[4], points[12])
    distances[5] = calculate_distance(points[5], points[11])
    distances[6] = calculate_distance(points[6], points[10])
    distances[7] = calculate_distance(points[7], points[9])


    # Border distances
    distances[8] = calculate_distance(points[0], points[1])
    distances[9] = calculate_distance(points[1], points[2])
    distances[10] = calculate_distance(points[2], points[3])
    distances[11] = calculate_distance(points[3], points[4])
    distances[12] = calculate_distance(points[4], points[5])
    distances[13] = calculate_distance(points[5], points[6])
    distances[14] = calculate_distance(points[6], points[7])
    distances[15] = calculate_distance(points[7], points[8])


    # Cheekbone distances

    distances[16] = calculate_distance(points[31], points[36])
    distances[17] = calculate_distance(points[32], points[41])
    distances[18] = calculate_distance(points[33], points[40])
    distances[19] = calculate_distance(points[33], points[39])
    distances[20] = calculate_distance(points[33], points[42])
    distances[21] = calculate_distance(points[33], points[47])
    distances[22] = calculate_distance(points[34], points[46])
    distances[23] = calculate_distance(points[35], points[45])


    # Height distances

    distances[24] = calculate_distance(points[4], points[17])
    distances[25] = calculate_distance(points[5], points[18])
    distances[26] = calculate_distance(points[6], points[19])
    distances[27] = calculate_distance(points[7], points[20])
    distances[27] = calculate_distance(points[8], points[21])
    distances[28] = calculate_distance(points[8], points[22])
    distances[29] = calculate_distance(points[9], points[23])
    distances[30] = calculate_distance(points[10], points[24])
    distances[31] = calculate_distance(points[11], points[25])
    distances[32] = calculate_distance(points[12], points[26])

    # Nasal Septum distances

    distances[32] = calculate_distance(points[27], points[31])
    distances[33] = calculate_distance(points[28], points[31])
    distances[34] = calculate_distance(points[29], points[31])
    distances[35] = calculate_distance(points[30], points[31])

    distances[37] = calculate_distance(points[27], points[32])
    distances[38] = calculate_distance(points[28], points[32])
    distances[39] = calculate_distance(points[29], points[32])
    distances[40] = calculate_distance(points[30], points[32])

    distances[41] = calculate_distance(points[27], points[33])
    distances[42] = calculate_distance(points[28], points[33])
    distances[43] = calculate_distance(points[29], points[33])
    distances[44] = calculate_distance(points[30], points[33])

    distances[45] = calculate_distance(points[27], points[34])
    distances[46] = calculate_distance(points[28], points[34])
    distances[47] = calculate_distance(points[29], points[34])
    distances[48] = calculate_distance(points[30], points[34])

    distances[49] = calculate_distance(points[27], points[35])
    distances[50] = calculate_distance(points[28], points[35])
    distances[51] = calculate_distance(points[29], points[35])
    distances[52] = calculate_distance(points[30], points[35])


    # Nose distances

    distances[53] = calculate_distance(points[27], points[36])
    distances[54] = calculate_distance(points[28], points[36])
    distances[55] = calculate_distance(points[29], points[36])
    distances[56] = calculate_distance(points[30], points[36])

    distances[57] = calculate_distance(points[27], points[39])
    distances[58] = calculate_distance(points[28], points[39])
    distances[59] = calculate_distance(points[29], points[39])
    distances[60] = calculate_distance(points[30], points[39])

    distances[61] = calculate_distance(points[27], points[42])
    distances[61] = calculate_distance(points[28], points[42])
    distances[63] = calculate_distance(points[29], points[42])
    distances[64] = calculate_distance(points[30], points[42])

    distances[65] = calculate_distance(points[27], points[45])
    distances[66] = calculate_distance(points[28], points[45])
    distances[67] = calculate_distance(points[29], points[45])
    distances[68] = calculate_distance(points[30], points[45])

    #to_normalize = calculate_distance(points[27], points[33])

    #distances[:] = [x / to_normalize for x in distances]

    return distances

