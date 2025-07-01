import numpy as np

# P0 + b2*P2 for angle theta 
def leg2(beta, theta):
    return 1 + beta*0.5 * (3*np.cos(theta)**2 - 1)

# simulate a pad (Px, Py, Pz) with N points
def padsim(N, p, dp, b2):
    
    Px = []
    Py = []
    Pz = []
    
    count = 1
    while count <= N:
        
        p1 = np.random.uniform(-1.1*p, 1.1*p)
        p2 = np.random.uniform(-1.1*p, 1.1*p)
        p3 = np.random.uniform(-1.1*p, 1.1*p)
        
        pr = np.sqrt(p1**2 + p2**2 + p3**2)
        if (pr < p+dp) and (p > p-dp):
            
            ang = np.arccos(p1 / pr)
        
            y1 = np.random.uniform(0,1)
            y2 = leg2(b2, ang) / leg2(b2, 0)
            
            if y1 <= y2:
                Px.append(p1)
                Py.append(p2)
                Pz.append(p3)
                count += 1
                
                if count % 100000 == 0: print("#{} / {:.0f}".format(count, N))
    
    return (Px, Py, Pz)