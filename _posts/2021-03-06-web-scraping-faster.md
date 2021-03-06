---
title: "Web scraping faster"
date: 2021-03-06
categories:
  - python
tags:
  - web scraping
  - programming
  - beautifulsoup
  - asynchronous requests
---

### Using *grequests* to scrape multiple pages

A common Python library for interacting with URLs is [*requests*][requests-docs]. When combining *requests* with a webscraping library, such as [*BeautifulSoup*][soup-docs], you are able to retrieve information inside the website's HTML or XML files.

The thing is that if you need to scrape not only a single webpage, but a significant quantity, *requests* may not be the most efficient way. One faster approach is using [*grequests*][grequests]. It allows you to make multiple asynchronous HTTP requests with the same ease as if you were using *requests*.

I'll show you an example to compare the performance of both methods. To reproduce the example, make sure you have *requests, grequests* and *BeautifulSoup* installed:

```bash
$ pip install requests
$ pip install grequests
$ pip install bs4
```

We will scrape the IMDB's top 250 movies [url][top250] and, for each movie, we will extract its title and director. Every movie has its own url and we will access them all to obtain the information we want. The first step, thus, is to retrieve the list of the 250 urls and, since we do this from a single website, we use *requests* for this job and BeautifulSoup to do the actual extraction:

```python
import requests
from bs4 import BeasutifulSoup


r = requests.get('https://www.imdb.com/chart/top?ref_=nv_mv_250')
soup = BeautifulSoup(r, 'lxml')
```

Using your browser's HTML inspector (access it pressing `F12`), you can find that each movie is in a `td` object with `class="titleColumn"`. 
![HTML inspector](/assets/images/imdb_top250_html_inspector.png 'HTML inspector')
*Mozilla's HTML inspector showing the IMDB's top 250 movies*

This is exactly what we are going to use to find the url for each movie. We are now looking for the link to the movie's website. Inside each element found, there is supposed to be a `href` value. This value appended to `www.imdb.com` gives us the movie url. They will have the format of `/title/tt*******`. Every element of `top250` has a `contents` attribute and looking inside it we find the `href` value we are searching for.


```python
top250 = soup.find_all('td', class_='titleColumn')
urls = []
for elem in top250:
    url = elem.contents[1]['href']
    urls.append(url)
```

We can start, now, our comparison. Let's begin with requesting one url at a time. I'm going to define an auxiliary function to extract and parse the information from each movie page. As before, I used the browser's HTML inspector to find the elements to use with *BeautifulSoup*. Somehow, when I open IMDB it always gives the movie titles in Portuguese, so I had to look for the original title. I split the title using `(` as a delimiter because in the HTML `meta` object, the `og:title` gives `movie title (movie year)`. The director information is in the object `div` in `class="credit_summary_item"`. After debugging to find where exactly they were located in the soup object, the result is:
```python

def get_title_director(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    title_elem = soup.find('meta', property="og:title")
    title = title_elem.attrs['content'].split('(')[0]
    director_elem = soup.find('div', class_="credit_summary_item")
    director = director_elem.contents[3].contents[0]
    return title, director

```
> **OBS:** When coding for real, good practice tells you to actually define these 'random' numbers and delimeters as constants at the begining of the code. For simplicity, they remain here.

```python
import time

# One by one
start_time = time.time()
for i, url in enumerate(urls):
    title, director = get_title_director('www.imdb.com' + url)
    if title and director:
        print(f"{i + 1} - {title}- {director}")

print(f"One by one takes: {time.time() - start_time:.2f} seconds")
```

Let's see how to use *grequests* before I show you the results I've got. Notice that we use the same `urls` list. Now, we will get the movies' information in batches of 10. But first, we change a bit our auxiliary function:

```python
def get_batch_title_director(self, batch):
        rs = (grequests.get(url) for url in batch)
        resp = grequests.map(rs)
        titles = []
        directors = []
        for i, r in enumerate(resp):
        	soup = BeautifulSoup(r.text, 'lxml')
        	title_elem = soup.find('meta', property="og:title")
        	title = title_elem.attrs['content'].split('(')[0]
        	director_elem = soup.find('div', class_="credit_summary_item")
        	director = director_elem.contents[3].contents[0]
            titles.append(title)
            directors.append(director)
        return titles, directors
```

> **OBS:** When coding for real, you can define a parsing function called by both `get_title_director` and `get_batch_title_director` to avoid repeating code.

Scraping in batches of 10 urls, then, can be done as follows:

```python
start_time = time.time()
for i in range(0, len(urls), cfg.BATCH):
    batch = ['https://www.imdb.com' + url for url in urls[i:i + 10]]
    titles, directors = get_batch_title_director(batch)
    if titles and directors:
        for j, movie_info in enumerate(zip(titles, directors)):
            print(f"{j + i + 1} - {movie_info[0]}- {movie_info[1]}")

print(f"In batches takes: {time.time() - start_time:.2f} seconds")
```

The result I've received when bundling all this code together was:

```bash
One by one takes: 296.04 seconds
In batches takes: 59.04 seconds
```

Pretty amazing, right? If you want to better understand why this difference of around 5 times happens, I recommend reading more about [concurrency][concurrency] and about [*gevent*][gevent], the library used by *grequests* to make concurrency happen.

You can check my full code for this inmy GitHub [repo][imdb-repo].

[requests-docs]: https://requests.readthedocs.io
[soup-docs]: https://beautiful-soup-4.readthedocs.io/en/latest/
[grequests]: https://github.com/spyoungtech/grequests
[top250]: https://www.imdb.com/chart/top?ref_=nv_mv_250
[gevent]: http://www.gevent.org/
[concurrency]: https://en.wikipedia.org/wiki/Concurrency_(computer_science)
[imdb-repo]: https://github.com/lnros/top-250-imdb-movies-scraper