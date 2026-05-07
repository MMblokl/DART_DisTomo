import random
from skimage import draw, morphology
import numpy as np
from PIL import Image
from os import mkdir
from os.path import isdir
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage import draw, morphology


# Create dir for phantoms
if not isdir("./phantoms/"):
  mkdir("./phantoms/")


# Define random seed for numpy as random module
random.seed(167)
np.random.seed(167)

# First phantom
# Generation parameters
resolution = 512 # Image resolution; rXr
shape_var = 25 # Maximum size of random shape sizes.
margin = 75 # Outer margin for generating shape locations. Larger margin is more empty space in between object and image border.
background_amt = 60 # Number of background, dark shapes
foreground_amt = 10 # Number of foreground, light shapes

# Create empty image
img = np.zeros([resolution, resolution])

# Create initial background shape comprising of dark ellipses.
for i in range(background_amt):
  # 50-50 chance to place a circle, otherwise an ellipse.
  match random.randint(0,1):
    case 0:  # Draw ellipse within margin of empty space
      rr, cc = draw.ellipse(random.randint(margin,resolution - margin), # Random location within margin
                            random.randint(margin,resolution - margin),
                            shape_var+random.randint(0, shape_var), # Random shape from (0, shape_var)
                            shape_var+random.randint(0, shape_var),
                            rotation=random.randint(0,360), # Random 0-360 degree rotation
                            shape=img.shape
                            )
    case _: # Draw circle
      rr, cc = draw.disk(
          (random.randint(margin,resolution - margin),
           random.randint(margin,resolution - margin)), # Random location within margin
          shape_var+random.randint(0, shape_var), # Random width of circle according to shape_var
          shape=img.shape
          )

  # Background images have half intensity
  img[rr, cc] = 120

# Morphological closing operation to make background shape a bit more filled out
morphology.closing(img, footprint=morphology.disk(7), out=img)

# Add *foreground_amt* random circles or ellispes in the foreground, within a much smaller margin, which is 1.8 times the *margin* variable.
margin = int(margin*1.8) # Reduce the margin to be within the original

# Same code as before, using *foreground_amt* and full intensity
for i in range(foreground_amt):
  match random.randint(0,1):
    case 0: # Draw ellipse
      rr, cc = draw.ellipse(random.randint(margin,resolution - margin),
                            random.randint(margin,resolution - margin),
                            shape_var+random.randint(0, shape_var),
                            shape_var+random.randint(0, shape_var),
                            rotation=random.randint(0,360),
                            shape=img.shape
                            )
    case _: # Draw circle
      rr, cc = draw.disk(
          (random.randint(margin,resolution - margin),
           random.randint(margin,resolution - margin)),
          shape_var+random.randint(0, shape_var),
          shape=img.shape
          )
  img[rr, cc] = 255

# Reshape image intensities to 0,255
img = np.clip(img, 0, 255).astype(np.uint8)

# Save to png
im = Image.fromarray(img)
im.save("./phantoms/phantom1.png")

# Second phantom:
# Make sure seed is set
random.seed(167)
np.random.seed(167)

# Parameters
resolution = 512
n_points = 100 # Number of points to draw voronoi neighbourhoods from.
margin = 40 # Margin of empty space to not generate lines in, used in point clipping.

img = np.zeros([resolution, resolution])

# Generate n random points to use for the voronoi.
x_positions = np.random.uniform(0, resolution, n_points)
y_positions = np.random.uniform(0, resolution, n_points)
points = np.column_stack((x_positions, y_positions)) # Stack into numpy ndarray

# Initial voronoi
v = Voronoi(points)
vor = voronoi_plot_2d(v) # Create an actual 2d objects from the voronoi

# Draw each vertex in the voronoi in the numpy image
for ridge in v.ridge_vertices:
  if -1 in ridge: # Non-existant vertex, which is ignored. This happens as scipy generated voronois in inf spacing on the outer neighbourhoods.
    continue
  # Get vertex not in inf space
  vertex = v.vertices[ridge]

  # Get coordinates and clip to defined resolution, minus the margin
  r0, c0 = vertex[0]
  r1, c1 = vertex[1]
  r0 = np.clip(r0, margin, resolution - margin)
  c0 = np.clip(c0, margin, resolution - margin)
  r1 = np.clip(r1, margin, resolution - margin)
  c1 = np.clip(c1, margin, resolution - margin)

  # Draw the line using the coordinates.
  rr, cc = draw.line(int(r0), int(c0), int(r1), int(c1))
  img[rr, cc] = 255


# Rescale to (0, 255)
img = np.clip(img, 0, 255).astype(np.uint8)

# Dilation to get rid of very small segments on clipped edges.
morphology.dilation(img, footprint=morphology.disk(7), out=img)

# Erosion to make the network more sponge-like, with thicker segments where the vertices connect.
morphology.erosion(img, footprint=morphology.star(3), out=img)
# Save to png
im = Image.fromarray(img)
im.save("./phantoms/phantom2.png")

# Make sure seed is set
random.seed(167)
np.random.seed(167)

# Third phantom
# Parameters
resolution = 512
margin = 120 # Number of pixels to add as padding between the border of the image, prevents the shapes from touching image borders
shape_var = 60 # The maximum size of the shapes drawn in random shape values
# Number of shapes for outer shape and core objects.
n_outer = 40
n_core = 80

# The intensities of the 3 drawn regions used for the phantom
outer_intensity = 150
inner_intensity = 220
core_intensity = 110

# Definine image.
img = np.zeros((resolution, resolution), dtype=np.float32)

# Create the outer region
mask_outer = np.zeros_like(img)
# Place n_outer random shapes to draw the region around.
for _ in range(n_outer):
    r = random.randint(margin, resolution - margin)
    c = random.randint(margin, resolution - margin)

    # 50-50 to generate either an ellipse or a circle
    if random.randint(0, 1) == 0:
        rr, cc = draw.ellipse(
            r, c,
            shape_var + random.randint(0, 40), # Random ellipse shape
            shape_var + random.randint(0, 40),
            rotation=np.deg2rad(random.randint(0, 180)),  # Random 180 degree rotation
            shape=img.shape
        )
    else:
        rr, cc = draw.disk(
            (r, c),
            shape_var + random.randint(0, 40), # Random circle diameter.
            shape=img.shape
        )
    mask_outer[rr, cc] = 1

# Closing with a disk to fill out any holes in between randomly placed shaped
mask_outer = morphology.binary_closing(mask_outer, morphology.disk(15))
img[mask_outer] = outer_intensity # Fill drawn shapes with defined intensity

# Create the inner region of higher contrast withing the outer region
mask_inner = morphology.binary_erosion(mask_outer, morphology.disk(35)) # Binary erosion of a copy of the outer region.
img[mask_inner] = inner_intensity # Fill inner region with defined intensity


# Add spots as darker gray regions in the light background
core = np.zeros_like(img)

# add n_core random disks within the shape.
for _ in range(n_core):
    r = random.randint(margin+20, resolution - margin-40) # Slightly decrease the margin to make sure none of the disks touch the border.
    c = random.randint(margin+20, resolution - margin-40)
    rr, cc = draw.disk(
        (r, c),
        random.randint(5, 15),
        shape=img.shape
    )
    core[rr, cc] = 1

# Multiply the defined core shapes with the inner region binary mask with the core object binary mask to remove those outside this region.
core = core * mask_inner
img[core == 1] = core_intensity

# Reshape to 0,255
img = np.clip(img, 0, 255).astype(np.uint8)

# Save to png
im = Image.fromarray(img)
im.save("./phantoms/phantom3.png")