This project demonstrates how to analyze and visualize time series data using a real-world dataset of hourly bicycle counts collected from Seattle’s Fremont Bridge. The dataset includes eastbound and westbound bike counts since 2012, and the goal is to uncover trends and patterns in bicycle usage over time.

The process begins by downloading the dataset from a public GitHub URL and loading it into a Pandas DataFrame. A new column called "Total" is added to represent the combined east and west counts for each timestamp. Summary statistics (mean, min, max, standard deviation) give a quick overview of the dataset’s distribution.

To explore patterns in the data, several visualizations are created:

    A raw time series plot of hourly data, which is too dense for interpretation.

    Weekly resampling of the data helps reveal seasonal trends, showing that bike usage increases in summer and decreases in winter.

    A 30-day rolling sum is used to smooth the data and capture long-term usage trends.

    Finally, using groupby on the time of day (ignoring date), the code shows average traffic at different times of day, highlighting peak usage hours.

This time series analysis showcases the importance of resampling, smoothing, and aggregation to extract meaningful insights from temporal data and reveals clear patterns in urban cycling behavior throughout the week and across seasons.
