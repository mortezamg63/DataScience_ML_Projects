This project visualizes California cities based on their geographic location, population, and area, using a scatter plot where each city is represented by a point. The color of each point reflects the city’s population (on a logarithmic scale), and the size of each point corresponds to the city's land area in square kilometers.

The process begins by downloading a dataset containing city-level data including latitude, longitude, total population, and total area. These values are then extracted into separate variables for plotting.

Using Matplotlib and Seaborn, a scatter plot is created with longitude and latitude as coordinates. The color (c) is determined by log-scaled population, and size (s) by area, allowing for intuitive visual encoding of two dimensions of data in one plot. A color bar legend shows population scale, and a custom area legend is manually constructed using empty scatter plots to illustrate what different point sizes represent in terms of land area.

The result is a geospatial visualization that simultaneously shows the distribution, population density, and land size of cities in California, offering a clear and aesthetically pleasing way to compare urban characteristics across the state.
