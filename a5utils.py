import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc

def bilinear_interpolate(image, x, y):     

    """
    This function takes a 2D array of values (e.g. a grayscale
    image) and a set of coordinate points (x,y) and returns 
    the interpolated value at the given coordinates.
    
    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing an image
    
    x : float array of length N
    y : float array of length N
    
    Returns
    -------
    values : float array of length N
        interpolated values at coordinates (x,y)
    """
    
    # compute the indices of the 4 neigbhors in 
    #the grid for each point in (x,y)
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1]-1)
    x1 = np.clip(x1, 0, image.shape[1]-1)
    y0 = np.clip(y0, 0, image.shape[0]-1)
    y1 = np.clip(y1, 0, image.shape[0]-1)

    # get the values of the 4 neighbors
    Ia = image[ y0, x0 ]
    Ib = image[ y1, x0 ]
    Ic = image[ y0, x1 ]
    Id = image[ y1, x1 ]

    # compute the interpolation weights
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # compute the weighted combination of neighbors
    values = Ia*wa + Ib*wb + Ic*wc + Id*wd

    return values


def display_movie(image_array):
        
    """
    This function creates an animation from an array of image frames.
    The result can be displayed in a jupyter notebook via:
      HTML(display_movie(image_array).to_jshtml())
    
    Parameters
    ----------
    image_array : array of images
         An array containing of images, each of which is either MxN or MxNx3
            
    Returns
    -------
    animation : FuncAnimation
        resulting animation object.
 
    """
    fig = plt.figure()
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    ani = animation.FuncAnimation(fig, animate, frames=len(image_array))

    rc('animation', html='jshtml')
    
    return ani

class SelectPoints:
    """
    Class that encapsulates allowing the user to click on a matplotlib
    axis and renders the points they clicked on
    """
    def __init__(self,ax,npoints):
        self.ax = ax
        self.pts, = ax.plot([0],[0],'r.')
        self.npoints = npoints
        self.xs = list()
        self.ys = list()
        self.cid = self.pts.figure.canvas.mpl_connect('button_press_event',self)
        self.toffset = 0.05*(ax.viewLim.x1-ax.viewLim.x0)

    def __call__(self, event):      
        #ignore clicks outside the figure axis
        if event.inaxes!=self.pts.axes: 
            return
        
         #otherwise record the click location and draw point on the plot
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.pts.set_data(self.xs,self.ys)
        self.ax.text(event.xdata+self.toffset,event.ydata,('%s'%len(self.xs)),bbox=dict(facecolor='red',alpha=0.3),verticalalignment='top')
        self.pts.figure.canvas.draw()
        
        #once we have npoints clicked, stop listening for
        #click events so that we don't accidentally add more 
        if (len(self.xs) >= self.npoints):
            self.pts.figure.canvas.mpl_disconnect(self.cid)

def select_k_points(ax,npoints):
    """
    Function to allow for interactive selection of points, displaying
    a number along side each point you click in order to make it easier
    to maintain correspondence.
    
    Parameters
    ----------
    ax : matplotlib figure axis
        Indicates the axis where you want to allow the user to select points
    npoints : int
        How many points are needed from the user
        
    Returns
    -------
    SelectPoints object
        Returns an object with fields xs and ys that contain the point 
        coordinates once the user clicks
        
    """

    ax.set_title(('click to select %d points' % npoints))
    selectpoints = SelectPoints(ax,npoints)
    plt.show()
    return selectpoints

