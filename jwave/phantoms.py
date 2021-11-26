from jwave.geometry import _circ_mask

def three_circles(N):
    mask1 = _circ_mask(N, 8, (50,50))
    mask2 = _circ_mask(N, 5, (80,60))
    mask3 = _circ_mask(N, 10, (64,64))
    p0 = 5.*mask1 + 3.*mask2 + 4.*mask3
    return p0