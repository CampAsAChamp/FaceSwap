import numpy as np
import matplotlib.pyplot as plt
import pickle

#part 2
from matplotlib.path import Path
from scipy.spatial import Delaunay
from a5utils import bilinear_interpolate

#part 2 demo for displaying animations in notebook
from IPython.display import HTML
from a5utils import display_movie

#part 4 blending
from scipy.ndimage import gaussian_filter


def get_transform(pts_source, pts_target):
    """
    This function takes the coordinates of 3 points (corners of a triangle)
    and a target position and estimates the affine transformation needed
    to map the source to the target location.


    Parameters
    ----------
    pts_source : 2D float array of shape 2x3
         Source point coordinates
    pts_target : 2D float array of shape 2x3
         Target point coordinates

    Returns
    -------
    T : 2D float array of shape 3x3
        the affine transformation
    """

    assert (pts_source.shape == (2, 3))
    assert (pts_source.shape == (2, 3))

    # your code goes here  (see lecture #16)
    #     print("Source Pts: ", pts_source)

    x1 = pts_source[0, 0]
    x2 = pts_source[0, 1]
    x3 = pts_source[0, 2]

    y1 = pts_source[1, 0]
    y2 = pts_source[1, 1]
    y3 = pts_source[1, 2]

    x1Prime = pts_target[0, 0]
    x2Prime = pts_target[0, 1]
    x3Prime = pts_target[0, 2]

    y1Prime = pts_target[1, 0]
    y2Prime = pts_target[1, 1]
    y3Prime = pts_target[1, 2]

    A = [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    B = [[x1Prime, x2Prime, x3Prime], [y1Prime, y2Prime, y3Prime], [1, 1, 1]]

    T = np.matmul(B, np.linalg.inv(A))

    return T


def apply_transform(T, pts):
    """
    This function takes the coordinates of a set of points and
    a 3x3 transformation matrix T and returns the transformed
    coordinates


    Parameters
    ----------
    T : 2D float array of shape 3x3
         Transformation matrix
    pts : 2D float array of shape 2xN
         Set of points to transform

    Returns
    -------
    pts_warped : 2D float array of shape 2xN
        Transformed points
    """

    xPts = pts[0]
    yPts = pts[1]
    onesRow = np.ones(xPts.shape)

    finalStack = np.stack((xPts, yPts, onesRow))

    multiplied = np.matmul(T, finalStack)

    NormalizedX = multiplied[0] / multiplied[2]
    NormalizedY = multiplied[1] / multiplied[2]

    pts_warped = np.stack((NormalizedX, NormalizedY))

    assert (T.shape == (3, 3))
    assert (pts.shape[0] == 2)

    return pts_warped


def warp(image, pts_source, pts_target, tri):
    """
    This function takes a color image, a triangulated set of keypoints
    over the image, and a set of target locations for those points.
    The function performs piecewise affine wapring by warping the
    contents of each triangle to the desired target location and
    returns the resulting warped image.

    Parameters
    ----------
    image : 3D float array of shape HxWx3
         An array containing a color image

    pts_src: 2D float array of shape 2xN
        Coordinates of N points in the image

    pts_target: 2D float array of shape 2xN
        Coorindates of the N points after warping

    tri: 2D int array of shape Ntrix3
        The indices of the pts belonging to each of the Ntri triangles

    Returns
    -------
    warped_image : 3D float array of shape HxWx3
        resulting warped image

    tindex : 2D int array of shape HxW
        array with values in 0...Ntri-1 indicating which triangle
        each pixel was contained in (or -1 if the pixel is not in any triangle)
    """
    assert (image.shape[2] == 3)  # this function only works for color images
    assert (tri.shape[1] == 3)  # each triangle has 3 vertices
    assert (pts_source.shape == pts_target.shape)
    assert (np.max(image) <= 1)  # image should be float with RGB values in 0..1

    ntri = tri.shape[0]
    (h, w, d) = image.shape

    # for each pixel in the target image, figure out which triangle
    # it fall in side of so we know which transformation to use for
    # those pixels.
    #
    # tindex[i,j] should contain a value in 0..ntri-1 indicating which
    # triangle contains pixel (i,j).  set tindex[i,j]=-1 if (i,j) doesn't
    # fall inside any triangle

    # checks whether a point falls inside a specified polygon.
    tindex = -1 * np.ones((h, w))
    xx, yy = np.mgrid[0:h, 0:w]
    pcoords = np.stack((yy.flatten(), xx.flatten()), axis=1)
    for t in range(ntri):
        # For the pts in the triangle get the x,y coordinates in the target
        Pair = pts_target[:, tri[t]]
        Pair = np.transpose(Pair)

        corners = Pair  # Vertices of triangle t.  Path expects a Kx2 array of vertices as input
        path = Path(corners)
        mask = path.contains_points(pcoords)
        mask = mask.reshape(h, w)

        # set tindex[i,j]=t any where that mask[i,j]=True
        tindex[np.where(mask[:, :] == True)] = t

    # compute the affine transform associated with each triangle that
    # maps a given target triangle back to the source coordinates
    np.set_printoptions(threshold=1000)
    Xsource = np.zeros((2, h * w))  # source coordinate for each output pixel
    tindex_flat = tindex.flatten()  # flattened version of tindex as an h*w length vector

    for t in range(ntri):
        # coordinates of target/output vertices of triangle t
        ptarg = pts_target[:, tri[t]]

        # coordinates of source/input vertices of triangle t
        psrc = pts_source[:, tri[t]]

        # compute transform from ptarg -> psrc
        T = get_transform(ptarg, psrc)

        # extract coordinates of all the pixels where tindex==t
        pcoords_t = np.argwhere(tindex == t)
        pcoords_t = np.transpose(pcoords_t)
        pcoords_t[[0, 1]] = pcoords_t[[1, 0]]  # Swap x row with y row as argwhere gives you y,x

        # store the transformed coordinates at the correspondiong locations in Xsource
        Xsource[:, tindex_flat == t] = apply_transform(T, pcoords_t)

    # now use interpolation to figure out the color values at locations Xsource
    warped_image = np.zeros(image.shape)
    warped_image[:, :, 0] = bilinear_interpolate(image[:, :, 0], Xsource[0, :], Xsource[1, :]).reshape(h, w)
    warped_image[:, :, 1] = bilinear_interpolate(image[:, :, 1], Xsource[0, :], Xsource[1, :]).reshape(h, w)
    warped_image[:, :, 2] = bilinear_interpolate(image[:, :, 2], Xsource[0, :], Xsource[1, :]).reshape(h, w)

    # clip RGB values outside the range [0,1] to avoid warning messages
    # when displaying warped image later on
    warped_image = np.clip(warped_image, 0., 1.)

    return (warped_image, tindex)




###################################
##
###################################
#
# Write some test cases for your affine_transform function
#

# check that using the same source and target should yield identity matrix
src = np.array([[1,2,3],[2,3,9]])
targ = np.array([[1,2,3],[2,3,9]])
print(get_transform(src,targ).astype(int))

# check that if targ is just a translated version of src, then the translation
# appears in the expected locations in the transformation matrix
src = np.array([[1,2,3],[2,3,9]])
targ = np.array([[2,4,6],[4,6,18]])
print(get_transform(src,targ).astype(int))

# random tests... check that for two random
# triangles the estimated transformation correctly
# maps one to the other
for i in range(5):
    src = np.random.random((2,3))
    targ = np.random.random((2,3))
    T = get_transform(src,targ)
    targ1 = apply_transform(T,src)
    assert(np.sum(np.abs(targ-targ1))<1e-12)
    print("Done with iteration ", i)


#######################################
##
#######################################
#
# Test your warp function
#

#make a color checkerboard image
(xx,yy) = np.mgrid[1:200,1:300]
G = np.mod(np.floor(xx/10)+np.floor(yy/10),2)
B = np.mod(np.floor(xx/10)+np.floor(yy/10)+1,2)
image = np.stack((0.5*G,G,B),axis=2)

#coordinates of the image corners
pts_corners = np.array([[0,300,300,0],[0,0,200,200]])

#points on a square in the middle + image corners
pts_source = np.array([[50,150,150,50],[50,50,150,150]])
pts_source = np.concatenate((pts_source,pts_corners),axis=1)

#points on a diamond in the middle + image corners
pts_target = np.array([[100,160,100,40],[40,100,160,100]])
pts_target = np.concatenate((pts_target,pts_corners),axis=1)

#compute triangulation using mid-point between source and
#target to get triangles that are good for both.
pts_mid = 0.5*(pts_target+pts_source)
trimesh = Delaunay(pts_mid.transpose())
#we only need the vertex indices so extract them from
#the data structure returned by Delaunay
tri = trimesh.simplices.copy()

# display initial image
plt.imshow(image)
plt.triplot(pts_source[0,:],pts_source[1,:],tri,color='r',linewidth=2)
plt.plot(pts_source[0,:],pts_source[1,:],'ro')
plt.show()

# display warped image
(warped,tindex) = warp(image,pts_source,pts_target,tri)
plt.imshow(warped)
plt.triplot(pts_target[0,:],pts_target[1,:],tri,color='r',linewidth=2)
plt.plot(pts_target[0,:],pts_target[1,:],'ro')
plt.show()

# display animated movie by warping to weighted averages
# of pts_source and pts_target

#assemble an array of image frames
movie = []
for t in np.arange(0,1,0.1):
    pts_warp = (1-t)*pts_source+t*pts_target
    print("Finished Warping Points")
    warped_image,tindex = warp(image,pts_source,pts_warp,tri)
#     print("Finished warped Image")
    movie.append(warped_image)

#use display_movie function defined in a5utils.py to create an animation
HTML(display_movie(movie).to_jshtml())


##################################
##
##################################
# load in the keypoints and images select_keypoints.ipynb
f = open('face_correspondeces.pckl', 'rb')
image1, image2, pts1, pts2 = pickle.load(f)
f.close()

# add the image corners as additional points so that the
# triangles cover the whole image
top = 0
left = 0
bottom = image1.shape[0]  # Height
right = image1.shape[1]  # Width
image1Corners = np.array([[left, left, right, right], [top, bottom, top, bottom]])

top = 0
left = 0
bottom = image2.shape[0]  # Height
right = image2.shape[1]  # Width
image2Corners = np.array([[left, left, right, right], [top, bottom, top, bottom]])

pts1Top = np.append(pts1[0], image1Corners[0])
pts1Bot = np.append(pts1[1], image1Corners[1])
pts1 = np.stack((pts1Top, pts1Bot))

# print("Pts1\n", pts1)

pts2Top = np.append(pts2[0], image2Corners[0])
pts2Bot = np.append(pts2[1], image2Corners[1])
pts2 = np.stack((pts2Top, pts2Bot))

# compute triangulation using mid-point between source and
# target to get triangles that are good for both.

pts_mid = 0.5 * (pts1 + pts2)
trimesh = Delaunay(pts_mid.transpose())

# we only need the vertex indices so extract them from
# the data structure returned by Delaunay
tri = trimesh.simplices.copy()

# generate the frames of the morph
i = 0
movie = []
for t in np.arange(0, 1, 0.05):
    pts_warp = (1 - t) * pts1 + pts2 * t
    warped_image1, tindex1 = warp(image1, pts1, pts_warp, tri)
    warped_image2, tindex2 = warp(image2, pts2, pts_warp, tri)
    warped_image = warped_image1 * (1 - t) + warped_image2 * t
    movie.append(warped_image)
    print("Finished with frame ", i)
    i = i + 1

# for t in np.arange(0,1,0.1):
#     pts_warp = (1-t)*pts_source+t*pts_target
#     warped_image,tindex = warp(image,pts_source,pts_warp,tri)
#     movie.append(warped_image)


# display original images and overlaid triangulation
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image1)
ax1.triplot(pts1[0, :], pts1[1, :], tri, color='g', linewidth=1)
ax1.plot(pts1[0, :], pts1[1, :], 'r.')
# ax1.plot(image1Corners[0,:],image1Corners[1,:],'b.')

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(image2)
ax2.triplot(pts2[0, :], pts2[1, :], tri, color='g', linewidth=1)
ax2.plot(pts2[0, :], pts2[1, :], 'r.')
# ax1.plot(image2Corners[0,:],image2Corners[1,:],'b.')

plt.show()

# display images at t=0.25, t=0.5 and t=0.75
#   i.e. visualize movie[5], movie[10],movie[15]
morphedFig = plt.figure()
ax3 = morphedFig.add_subplot(1, 3, 1).imshow(movie[5])
ax4 = morphedFig.add_subplot(1, 3, 2).imshow(movie[10])
ax5 = morphedFig.add_subplot(1, 3, 3).imshow(movie[15])
morphedFig.show()

# optional: display as an animated movie
HTML(display_movie(movie).to_jshtml())

###############################
##
###############################
f = open('face_correspondeces.pckl','rb')
image1,image2,pts1,pts2 = pickle.load(f)
f.close()

#compute triangulation using mid-point between source and
#target to get triangles that are good for both images.
pts_mid = 0.5*(pts1+pts2)
trimesh = Delaunay(pts_mid.transpose())

#we only need the vertex indices so extract them from
#the data structure returned by Delaunay
tri = trimesh.simplices.copy()

# put the face from image1 in to image2
#     pts_warp = (1-t)*pts_source+t*pts_target
#     warped_image,tindex = warp(image,pts_source,pts_warp,tri)

#     pts_warp = (1-t)*pts1 + pts2*t
#    warped_image1,tindex1 = warp(image1, pts1, pts_warp, tri)

(warped,tindex) = warp(image1,pts1,pts2, tri)
# Make a mask which is true inside the face region and false everywhere else
mask = np.where(tindex == -1, 0, 1)

alpha = gaussian_filter(mask, sigma=1, output='float64')                        # Blends the entire mask
ZeroAreas = np.argwhere(mask == 0)                      # Get coordinates where the mask is zero
alpha[ZeroAreas[::,0], ZeroAreas[::,1]] = 0               # Set alphas to zero where mask is zero
alpha[alpha != 1] -= np.min(alpha[np.nonzero(alpha)])

# Clips alphas again
ZeroAreas = np.argwhere(mask == 0)                      # Get coordinates where the mask is zero
alpha[ZeroAreas[::,0], ZeroAreas[::,1]] = 0               # Set alphas to zero where mask is zero

alpha = alpha*mask
alpha = np.divide(alpha, np.max(alpha))

alpha1 = alpha

swap1 = np.zeros(image1.shape)
swap2 = np.zeros(image2.shape)

# do an alpha blend of the warped image1 and image2
swap1[:,:,0] = alpha*warped[:,:,0]+(np.ones(alpha.shape)-alpha) * image2[:,:,0]
swap1[:,:,1] = alpha*warped[:,:,1]+(np.ones(alpha.shape)-alpha) * image2[:,:,1]
swap1[:,:,2] = alpha*warped[:,:,2]+(np.ones(alpha.shape)-alpha) * image2[:,:,2]


####################################
# SECOND IMAGE
####################################
(warped,tindex) = warp(image2,pts2,pts1, tri)
# Make a mask which is true inside the face region and false everywhere else
mask = np.where(tindex == -1, 0, 1)

alpha = gaussian_filter(mask, sigma=1, output='float64')                        # Blends the entire mask
ZeroAreas = np.argwhere(mask == 0)                      # Get coordinates where the mask is zero
alpha[ZeroAreas[::,0], ZeroAreas[::,1]] = 0               # Set alphas to zero where mask is zero
alpha[alpha != 1] -= np.min(alpha[np.nonzero(alpha)])

# Clips alphas again
ZeroAreas = np.argwhere(mask == 0)                      # Get coordinates where the mask is zero
alpha[ZeroAreas[::,0], ZeroAreas[::,1]] = 0               # Set alphas to zero where mask is zero

alpha = alpha*mask
alpha = np.divide(alpha, np.max(alpha))
alpha2 = alpha


#now do the swap in the other direction
swap2[:,:,0] = alpha*warped[:,:,0]+(np.ones(alpha.shape)-alpha) * image1[:,:,0]
swap2[:,:,1] = alpha*warped[:,:,1]+(np.ones(alpha.shape)-alpha) * image1[:,:,1]
swap2[:,:,2] = alpha*warped[:,:,2]+(np.ones(alpha.shape)-alpha) * image1[:,:,2]

plt.rcParams['figure.figsize'] = [9,9]

# display the images with the keypoints overlayed
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(image1)
ax1.triplot(pts1[0,:],pts1[1,:],tri,color='g',linewidth=1)
ax1.plot(pts1[0,:],pts1[1,:],'r.')
# ax1.plot(image1Corners[0,:],image1Corners[1,:],'b.')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(image2)
ax2.triplot(pts2[0,:],pts2[1,:],tri,color='g',linewidth=1)
ax2.plot(pts2[0,:],pts2[1,:],'r.')
# ax1.plot(image2Corners[0,:],image2Corners[1,:],'b.')

plt.show()

# display the face swapping result
fig2 = plt.figure()
fig2.add_subplot(3,2,1).imshow(image1)
fig2.add_subplot(3,2,2).imshow(image2)
fig2.add_subplot(3,2,3).imshow(swap1)
fig2.add_subplot(3,2,4).imshow(swap2)
# fig2.add_subplot(3,3,7).imshow(alpha1, cmap=plt.cm.gray)
# fig2.add_subplot(3,3,8).imshow(alpha2, cmap=plt.cm.gray)

# fig2.add_subplot(2,2,4).imsh(swap2)


fig2.show()


