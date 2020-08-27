---
title: "Django database optimisation"
date: 2020-08-27T20:00:00
categories:
  - blog
tags:
  - Django
  - Python
  - Optimizations
---


The Django ORM makes interacting with the database a breeze, but without due care can also lead to poor performance.


# Example models


Let's say we have 2 very simple models, an `Artist`and her `Artwork`:

```python
class Artist(models.Model):
    name = models.CharField(max_length=40, blank=False)

class Artwork(models.Model):
    artist = models.ForeignKey(Artist, related_name='artworks', on_delete=models.CASCADE)
    title = models.CharField(max_length=254,  blank=True)
```

# Fetching artworks for each artist

Imagine we had a user listing endpoints (for example in a Django rest framework API), where we serialized each user and a nested array of their artworks

```python
artists = Artist.objects.all()

for artist in artists:
    artwork = artist.artworks.first()
    if artwork is not None:
        print(artwork.title)
```

How many database queries would you expect?

If we tried this in the `python manage.py shell --print-sql`, you would see `N+1` queries, where `N` is the number of artists in our database.

```sql
SELECT 
       "artist"."name",
       "artist"."id",
FROM "artist"

SELECT
       "artwork_artwork"."id",
       "artwork_artwork"."title",
FROM "artwork_artwork"
WHERE ("artwork_artwork"."artist_id" = '5e9eceb7-5d4e-419a-a5e6-cb96b6e1fcca'::uuid)
 LIMIT 1


SELECT "artwork_artwork"."id",
       "artwork_artwork"."title",
FROM "artwork_artwork"
WHERE ("artwork_artwork"."artist_id" = '069d5b3c-3fa4-4236-a054-13b73183ac49'::uuid)
 LIMIT 1

.
.
.
```

Here first we have the 1 `Artist` lookup and then the ensuing `N` related `Artwork` lookups (another database hit per artist)

## Enter prefetch related

This `N+1` problem is so common that Django gave us the `prefetch_related` queryset method


If we instead do

```python
artists = Artist.objects.prefetch_related('artworks').all()

for artist in artists:
    artwork = artist.artworks.first()
    if artwork is not None:
        print(artwork.title)
```

and observe the SQL queries in the shell, we'd now see

```sql

SELECT 
       "artist"."name",
       "artist"."id",
FROM "artist"


SELECT "artwork_artwork"."id",
       "artwork_artwork"."title"
FROM "artwork_artwork"
 WHERE ("artwork_artwork"."artist_id" IN ('5e9eceb7-5d4e-419a-a5e6-cb96b6e1fcca'::uuid, '069d5b3c-3fa4-4236-a054-13b73183ac49'::uuid, 'b202cfbd-c34b-4453-8d96-d00c314655c1'::uuid, 'dc1497c4-cf71-4b29-bcd4-dc1ba1836f3f'::uuid, '5f2b5604-f204-4130-a7a4-529f435a11cc'::uuid))
```

So now we have just 2 SQL queries! No matter how many users in our database. This is a lot better than `N+1`

Django will execute the second query and store the results on `Artist` queryset, so any future lookup with `artist.artwork` will happen in Python without hitting the database.

The downside of this is that all this data will now be stored in memory, so care is needed that this doesn't grow too large.


## Select related

Conversely, if instead of a `1-N` relationship, we were looking at fetching a related model of which there was only one, for example

```python
artworks = Artwork.objects.all()
for artwork in artworks:
    print(f'{artwork.title} is by {artwork.artist.name})
```

Then once again, we'd have an `N+1` issue: 1 query to grab the artworks from the database, and then 1 query per each of the `N` artworks to fetch the parent `Artist`:


```sql
SELECT "artwork_artwork"."id",
       "artwork_artwork"."title",
FROM "artwork_artwork"

SELECT "artist"."id",
    
       "artist"."name",
FROM "artist"
WHERE "artist"."id" = '034544ed-a262-4c86-a061-47891daf2824'::uuid

.
.
.
```

To avoid this issue, Django provides us with `select_related`


```python
artworks = Artwork.objects.select_related('artist').all()
 ```

 which will do a SQL JOIN to reduce all those queries to just a single SQL query before caching it on the Python queryset:

 ```sql

 SELECT 
       "artwork_artwork"."id",
       "artwork_artwork"."title",
       "artist"."id",
       "artist"."_name",
  FROM "artwork_artwork"
 INNER JOIN "artist"
    ON ("artwork_artwork"."artist_id" = "artist"."id")
```

# Use with DRF

Whenever you use a nested serializer in Django-rest-framework, you run the risk of this `N+1` problem, so it's worth ensuring you are not hitting the database hard.

One way to implement the above methods would be to override the `get_queryset` method of the viewset


```python
class ArtistViewSet(viewsets.ModelViewSet):

    serializer_class = ArtistSerializer

    def get_queryset(self):
        # The base user queryset
        # Without the prefetch_related we'd hit a N+1 with a fetch of those things per user rather than batch/cache
        qs = Artist.objects.select_related('auth_token')\
            .prefetch_related('artworks')
        return qs
```


# Monitoring performance with Django


### Runserver plus and shell plus

If you use [Runserverplus](https://django-extensions.readthedocs.io/en/latest/runserver_plus.html) then you develop locally with the `--print-sql` switch, which is very useful when trying to check what database queries are being made by each view. Simiarly with `shell_plus --print-sql`

You could also commit to logging all slow SQL queries with a logging filter

```python
# settings.py

SLOW_SQL_THRESHOLD = 0.001

class SlowQueriesFilter(logging.Filter):
    """Filter slow queries"""

    def filter(self, record):
        duration = record.duration
        if duration > SLOW_SQL_THRESHOLD:
            return True
        return False


LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'slow_queries': {
            '()': SlowQueriesFilter,
        },
    },
.
.
.
    'loggers': {
        'django.db.backends': {
            'handlers': ['console'],
            'filters': ['slow_queries'],
            'level': 'DEBUG'
        }
    }
}
```


Another useful tool is [Django-debug-toolbar](https://django-debug-toolbar.readthedocs.io/en/latest/), which will show you how many SQL queries and a breakdown of them for each view.

The downside is that this only works in a browser, but for `GET` API endpoints, if you execute them in a browser you can still use this tool for performance insights.