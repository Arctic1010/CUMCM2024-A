#问题2 碰撞时刻板凳龙示意图

import matplotlib.pyplot as plt
import numpy as np
pi = 3.1415926535
head_len = 3.41 - 0.55
body_len = 2.2 - 0.55
crash_time = 412.47383767112257

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

def arc_len(mid):
    a = 0.55 / 2 / pi
    return a / 2 * (mid * np.sqrt(mid * mid + 1) + np.log(mid + np.sqrt(mid * mid + 1)))

def point_distance(x,y,a,b):
    return np.sqrt((x-a)*(x-a)+(y-b)*(y-b))

def get_nxt_point(x,y,otheta,len):
    theta = otheta
    l = 0
    r = theta
    a = 0.55 / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(theta) - arc_len(mid) < len):
            r = mid
        else:
            l = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_prev_point(x,y,otheta,lenn):
    theta = otheta
    l = theta 
    r = theta + 2 * pi
    a = 0.55 / 2 / pi
    for _ in range(50):
        mid = (l + r) / 2
        if(arc_len(mid) - arc_len(theta) < lenn):
            l = mid
        else:
            r = mid
    ntheta = l
    nrho = a * ntheta
    nx, ny = Polar_to_Rectangular(nrho, ntheta)
    return nx, ny, ntheta

def get_prev_board(x,y,otheta,len):
    theta = otheta
    l = theta 
    r = theta + pi
    a = 0.55 / 2 / pi
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

def draw_rectangle(recta,color1):
    x = [recta.p[0].x, recta.p[1].x, recta.p[2].x, recta.p[3].x,recta.p[0].x]
    y = [recta.p[0].y, recta.p[1].y, recta.p[2].y, recta.p[3].y,recta.p[0].y]
    plt.plot(x,y,linewidth = 3, color = color1)
plt.figure(figsize=(6, 6)) 
line_x = []
line_y = []
for i in range(5000):
    atheta = 12.5 * pi / 5000 * i
    arho = atheta * 0.55 / 2 / pi
    ax, ay = Polar_to_Rectangular(arho, atheta)
    line_x.append(ax)
    line_y.append(ay)
plt.plot(line_x,line_y)

start_point = point(8.8,0)
start_theta = 32 * pi

x,y,theta = get_nxt_point(start_point.x, start_point.y, start_theta, crash_time)
tx,ty,ttheta = get_prev_board(x,y,theta,head_len)
plt.plot(x,y,'o')
plt.plot(tx,ty,'o')
draw_rectangle(get_rectangle(x,y,tx,ty),'blue')

for i in range(18):
    x,y,theta = get_prev_board(tx,ty,ttheta,body_len)
    plt.plot(x,y,'o')
    draw_rectangle(get_rectangle(x,y,tx,ty),'red')
    tx = x
    ty = y
    ttheta = theta
plt.savefig("plot_2.eps", dpi=300)
plt.show()
