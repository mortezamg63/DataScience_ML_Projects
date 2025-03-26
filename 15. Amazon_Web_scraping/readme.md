This code implements a web scraping project that targets Amazon India's Best Sellers page in the Teaching & Education books category. The goal is to extract information about the top 50 bestselling books, specifically focusing on the author names and book ratings.

The workflow begins by setting up a base URL and HTTP headers (including a User-Agent) to mimic a real browser and avoid being blocked by Amazon's server. It then iterates through the first three pages of the bestseller list, using the requests library to fetch HTML content and BeautifulSoup to parse it. Within each page, it locates the relevant HTML elements containing author names and rating info, collecting this data into a list of dictionaries while making sure not to exceed 50 entries.

After the data collection, the project wraps up by converting the list into a Pandas DataFrame and exporting it as a CSV file for further analysis or visualization. The result is a neatly structured file containing details of the top 50 books, which can be useful for market research, educational analysis, or trend monitoring.

In summary, this project is a practical demonstration of web scraping, data parsing, and data storage, using real-world e-commerce data and Python libraries like requests, BeautifulSoup, and pandas.
