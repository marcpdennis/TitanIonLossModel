from scipy import interpolate
import numpy as np
from numpy import where
import matplotlib.pyplot as plt
from math import sin
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Wedge
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import os.path
import math as m
import pandas as pd

#Open data file and store in a data frame
data = pd.read_table('field_line_data.txt', delim_whitespace=True,header = None)
data.columns = ["x", "y", "z",'Bx','By','Bz','E_dens','E_temp','N2_dens','CH4_dens','N2+_rate','CH4+_rate']

#Create individual variables for each position
x = data['x']
y = data['y']
z = data['z']

rad=np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
alt=(rad-1.)*2575.

Bx = data['Bx']
By = data['By']
Bz = data['Bz']


#Create variable for magneitic field stength
totalb = np.sqrt(np.power(Bx,2) + np.power(By,2) + np.power(Bz,2))

#Create variable for electron density and temperature
E_dens = data['E_dens']
E_dens = E_dens*1e6    #convert from 1/cc to 1/m^3

E_temp = data['E_temp']

#Calculate distance between each point
distance = np.zeros(len(x))
s = np.zeros(len(x))

for k in range(1,len(x)):
    
    if (k == 0):
        distance[k] = 0
    else:
        distance[k] = m.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2 + (z[k] - z[k-1])**2)
        s[k]=s[k-1]+distance[k]
    

#Create sphere to represent Titan
N=200
stride=1
u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)
circlex = np.outer(np.cos(u), np.sin(v))
circley = np.outer(np.sin(u), np.sin(v))
circlez = np.outer(np.ones(np.size(u)), np.cos(v))


#Plot field line with magnetic field magnitude as the color
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(circlex,circley,circlez, linewidth=0.0, cstride=stride, rstride=stride)
ax.scatter(x+1,y+1,z+1,c=totalb)
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-3, 3)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
p = ax.scatter(x+1,y+1,z+1,c=totalb)
fig.colorbar(p, label = 'Magneitic Field Magnitude')
plt.subplots_adjust(right=2)
plt.show()



init_func=1   # Select stair case function (0) or sin^2 function (1)

# function defining the initial condition
if (init_func==0):
    def f(x):
        """Assigning a value of 1.0 for values less than 0.1"""
        f = np.zeros_like(x)
        f[np.where(x <= 0.1)] = 1.0
        return f
elif(init_func==1):
    def f(x):
        """A smooth sin^2 function between x_left and x_right"""
        f = np.zeros_like(x)
        x_left = 0.25
        x_right = 0.75
        xm = (x_right-x_left)/2.0
        f = where((x>x_left) & (x<x_right), np.sin(np.pi*(x-x_left)/(x_right-x_left))**4,f) 
        return f

 #Calculates the Q value at that time and u values, will pass out a scalar
def q(dt,x):
    return 0.1



# Lax-Friedrich Flux formulation 
def lax_friedrich_Flux(u,flux,source,t):
    u[1:-1] = (u[:-2] +u[2:])/2.0 -  dt*(flux[2:]-flux[:-2])/(2.0*dx)          
    + dt*(source[:-2] + source[2:])/2.0
    return u[1:-1]



# Lax-Friedrich Advection 
def lax_friedrich(u,flux,source):
    u[1:-1] = (u[:-2] +u[2:])/2.0 -  c*(u[2:] - u[:-2])/2.0
    return u[1:-1] 

#I decided to code this in to give us a smoother electron density profile
#eventually the code will include chemistry
def chapman_model(alt):
    alt0=1150  #peak of electron density
    ne0=1000   #peak electron density
    H=70
    z[:]=(alt[:]-alt0)/H
    ne=np.zeros(len(alt))
    ne[:]=ne0*np.exp(.5*(1-z[:]-np.exp(-z[:])))
    return ne


# Constants and parameters
a = 1.0 # wave speed
#tmin, tmax = 0.0, 1.0 # start and stop time of simulation
tmin, tmax = 0.0, 1.0 # start and stop time of simulation


Nx = len(s) # number of spatial points
c = 0.5 # courant number, need c<=1 for stability

# Discretize
#x = np.linspace(xmin, xmax, Nx+1) # discretization of space
dx = s[1]-s[0]
dt = c/1*dx # stable time step calculated from stability requirement
dt=0.0001
Nt = int((tmax-tmin)/dt) # number of time steps
time = np.linspace(tmin, tmax, Nt) # discretization of time


u = np.zeros((3,len(x)))
flux = np.zeros((3,len(x)))
source = np.zeros((3,len(x)))
source = np.zeros((3,len(x)))
flux= np.zeros((3,len(x)))
dlnAds = np.zeros(len(x))
dpeds = np.zeros(len(x))
rhoi=np.zeros(len(s))
pressi=np.zeros(len(s))
ui=np.zeros(len(s))
epsiloni=np.zeros(len(s))


A = 1/totalb
mi=28*1.6e-27  #mass of ion, kg
Ti=150
kb=1.38e-23    #Boltzman constant standard units
echarg = 1.9e-19

#Calculating some initial values
E_dens=chapman_model(alt)*1e6
rhoi=E_dens*mi
pressi=E_dens*kb*Ti
ui=np.zeros(len(x))
epsiloni = 0.5*A*rhoi*np.power(ui,2)+1/(1.5-1)*A*pressi

u[0,:] = A[:]*rhoi[:]
u[1,:] = A[:]*rhoi[:]*ui[:]
u[2,:] = A[:]*epsiloni[:]

#Central Differencing
for i in range(1,len(s)-1):
    dlnAds[i]=(np.log(A[i+1])-np.log(A[i-1]))/(s[i+1]-s[i-1])
    
#Forward Differencing
dlnAds[0]=(np.log(A[1])-np.log(A[0]))/(s[1]-s[0])

#Backwards Differencing
lst=len(s)-1
dlnAds[lst]=(np.log(A[lst])-np.log(A[lst-1]))/(s[lst]-s[lst-1])
    

#I plotted some of these initial values for debugging

#plt.plot(E_dens,alt)
#plt.xlabel('E_dens')
#plt.ylabel('Altitude')
#plt.show()        

#plt.plot(rhoi,alt)
#plt.xlabel('rhoi')
#plt.ylabel('Altitude')
#plt.show()


#plt.plot(ui,alt)
#plt.xlabel('ui')
#plt.ylabel('Altitude')
#plt.show()
   

#I simplified this a bit, for debugging
for i in range(200):
    
    #dt = max(.001,c/abs(max(ui))*dx)
   #I made the time step really small above, eventually it should depend on the velocities in the code
    
    t=time[i]
    
    #Extract relevant quantities from the results
    rhoi[:]=u[0,:]/A[:]
    ui[:]=u[1,:]/(A[:]*rhoi[:])
    pressi=rhoi/mi*kb*Ti
    epsiloni = 0.5*A*rhoi*np.power(ui,2)+1/(1.5-1)*A*pressi
    
      
    #Calculate electron pressure
    presse=rhoi/mi*kb*500
    
    #Central Differencing
    for i in range(1,len(s)-1):
        dpeds[i]=(presse[i+1]-presse[i-1])/(s[i+1]-s[i-1])
    
    #Forward Differencing
    dpeds[0]=(presse[1]-presse[0])/(s[1]-s[0])

    #Backwards Differencing
    dpeds[lst]=(presse[lst]-presse[lst-1])/(s[lst]-s[lst-1])
    
    #Calculate electric field
    Epar=-1/(echarg*rhoi/mi)*dpeds
    
    
    #Load flux and source terms
    flux[0,:]=A[:]*rhoi[:]*ui[:]
    flux[1,:]=A[:]*rhoi[:]*np.power(ui[:],2)+A[:]*pressi[:]
    flux[2,:]= A[:]*epsiloni[:]
    
    
    source[0,:]=0
    source[1,:]=A[:]*pressi[:]*dlnAds[:]-A[:]*rhoi*(echarg/mi)*Epar[:]
    source[2,:]=A[:]*rhoi[:]*ui[:]*(echarg/mi)*Epar[:]
    
    
    for k in range(len(u)):
        
        u_bc = interpolate.interp1d(s[-2:], u[k,-2:]) # interplate at right bndry
    
        u[k,1:-1] = lax_friedrich_Flux(u[k,:],flux[k,:],source[k,:],t) # calculate numerical solution of interior
        
        #This could be coded better, but I did a constant slope boundary condition at the top boundary
        u[k,len(s)-1] = (u[k,len(s)-2]-u[k,len(s)-3])/(s[len(s)-2]-s[len(s)-3])*(s[len(s)-1]-s[len(s)-2])+u[k,len(s)-2]
        
        #And a zero-slope at the bottom boundary
        u[k,0] = u[k,1]
    

#Plotting some of the various quantities
#The look of these plot (labels,etc) should be improved
        
plt.plot(E_dens,alt, c = 'aqua')
plt.xlabel('Electron Density ($cm^{-3}$)')
plt.ylabel('Altitude (km)')
plt.show()        

plt.plot(rhoi,alt, c = 'red')
plt.xlabel('Mass Density (kg / $m^3$)')
plt.ylabel('Altitude (km)')
plt.show()


plt.plot(ui,alt, c = 'green')
plt.xlabel('Speed of Ion Outflow')
plt.ylabel('Altitude (km)')
plt.show()

plt.plot(Epar,alt, c = 'indigo')
plt.xlabel('Magnitude of Electric Field Parallel to Magnetic Field Line (V/m)')
plt.ylabel('Altitude (km)')
plt.show()

plt.plot(A,alt, c = 'gold')
plt.xlabel('Magnetic Flux Tube Cross Sectional Area ($nT^{-3}$)')
plt.ylabel('Altitude (km)')
plt.show()

plt.plot(pressi,alt, c = 'peru')
plt.xlabel('Electron Pressure (atm)')
plt.ylabel('Altitude (km)')
plt.show()

plt.plot(epsiloni,alt, c = 'black')
plt.xlabel('Energy (J)')
plt.ylabel('Altitude (km)')
plt.show()
