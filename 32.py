import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
pi = 3.1415926535

class point:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y

class rect:
    p = [point(0,0),point(0,0),point(0,0),point(0,0),point(0,0)]

    def __init__(self,a,b,c,d,e,f,g,h):
        self.p = [point(a,b),point(c,d),point(e,f),point(g,h),point(a,b)]
        
    def print_point(self):
        print("Rect{")
        for i in range(4):
            print(f"({self.p[i].x},{self.p[i].y})")
        print("}")
class vec:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y

def normalize(v):
    len = np.sqrt(v.x*v.x+v.y*v.y)
    return vec(v.x / len, v.y / len)

def Rectangular_to_Polar(x, y): 
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan(y / x))
    return r, theta

def Polar_to_Rectangular(r, theta): 
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def arc_len(mid,s):
    a = s / 2 / pi
    return a / 2 * (mid * np.sqrt(mid * mid + 1) + np.log(mid + np.sqrt(mid * mid + 1)))

def point_distance(x,y,a,b):
    return np.sqrt((x-a)*(x-a)+(y-b)*(y-b))

def get_nxt_point(x,y,otheta,len,s):
    theta = otheta
    l = 0
    r = theta
    a = s / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(theta,s) - arc_len(mid,s) < len):
            r = mid
        else:
            l = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_prev_point(x,y,otheta,lenn,s):
    theta = otheta
    l = theta 
    r = theta + 2 * pi
    a = s / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(mid,s) - arc_len(theta,s) < lenn):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_prev_board(x,y,otheta,len,s):
    theta = otheta
    l = theta 
    r = theta + pi
    a = s / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        tx, ty = Polar_to_Rectangular(a * mid, mid)
        if(point_distance(x, y, tx, ty) < len):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_rectangle(x,y,a,b):
    line_vec = normalize(vec(x-a,y-b))
    up_vec = normalize(vec(line_vec.y, -line_vec.x))
    
    point1_x = x + line_vec.x * 0.275 + up_vec.x * 0.15
    point1_y = y + line_vec.y * 0.275 + up_vec.y * 0.15
    
    point2_x = x + line_vec.x * 0.275 + up_vec.x * -0.15
    point2_y = y + line_vec.y * 0.275 + up_vec.y * -0.15
    
    point3_x = a + line_vec.x * -0.275 + up_vec.x * -0.15
    point3_y = b + line_vec.y * -0.275 + up_vec.y * -0.15
    
    point4_x = a + line_vec.x * -0.275 + up_vec.x * 0.15
    point4_y = b + line_vec.y * -0.275 + up_vec.y * 0.15
    
    return rect(point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y)

def quick_judge(a,b,c,d):
    if (max(a.x,b.x) < min(c.x,d.x) or
        max(c.x,d.x) < min(a.x,b.x) or
        max(a.y,b.y) < min(c.y,d.y) or
        max(c.y,d.y) < min(a.y,b.y)) :
        return False
    else:
        return True

def xmult(a,b,c,d):
    vectorAx = b.x - a.x
    vectorAy = b.y - a.y
    vectorBx = d.x - c.x
    vectorBy = d.y - c.y
    return (vectorAx * vectorBy - vectorAy * vectorBx)
def cross(a,b,c,d):
    if not quick_judge(a,b,c,d):
        return False
    xmult1 = xmult(c,d,c,a)
    xmult2 = xmult(c,d,c,b)
    xmult3 = xmult(a,b,a,c)
    xmult4 = xmult(a,b,a,d)
    if xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0:
        return True
    else:
        return False

def collide_detect(s,t):
    for i in range(4):
        for j in range(4):
            if(cross(s.p[i], s.p[i + 1], t.p[j], t.p[j + 1])):
                return True
    return False

def check(s):
    t_prev = 2000
    data = np.zeros((448))
    x = s * 16
    y = 0
    theta = 32 * pi
    org_x = x
    org_y = y
    org_theta = theta
    rect_array = []
    for t in range(223 * 10):
        if(t != 0):
            x = org_x
            y = org_y
            theta = org_theta
            if(point_distance(x,y,0,0)<4.5):
                print(f"{s} OK, True, t = {t}")
                return True
            x,y,theta = get_nxt_point(x,y,theta,0.1,s)
            org_x = x
            org_y = y
            org_theta = theta
        if(t > t_prev):
            data[0] = x
            data[1] = y
        
            x,y,theta = get_prev_board(x,y,theta,3.41 - 0.55,s)
            data[2] = x
            data[3] = y
            rect_array.append(get_rectangle(data[0],data[1],data[2],data[3]))
            for i in range(222):
                x,y,theta = get_prev_board(x,y,theta, 2.2 - 0.55,s)
                data[(i + 2) * 2] = x
                data[(i + 2) * 2 + 1] = y
                rect_array.append(get_rectangle(data[(i+2)*2-2],data[(i+2)*2-1],data[(i+2)*2],data[(i+2)*2+1]))
            for i in range(len(rect_array)):
                for j in range(i+2, len(rect_array)):
                    if(collide_detect(rect_array[i],rect_array[j])):
                        print(f"{s} collided, False, t = {t}")
                        rect_array[i].print_point()
                        rect_array[j].print_point()
                        
                        return False
            rect_array.clear()
    return False
L = 0.4402
R = 0.4500

# L = 0.4502
# R = 0.4503 -> Now binary searching: L = 0.45029990234374995, R = 0.4503

for _ in range(30):
    mid = (L + R) / 2
    print(f"Now binary searching: L = {L}, R = {R}")
    if(check(mid)):
        R = mid
    else:
        L = mid

print(L)


# import pandas as pd
# df = pd.DataFrame(data_xlsx)
# df.to_excel("output2.xlsx",index=False)
# print("Done.")
