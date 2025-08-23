'''
PART 1: ETL the dataset and save in `data/`

Here is the imbd_movie data:
https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true

It is in JSON format, so you'll need to handle accordingly and also figure out what's the best format for the two analysis parts. 
'''

#import os
import pandas as pd
import json
from pathlib import Path
from urllib.request import urlopen
import hashlib

DATA_DIR = Path('data')
SOURCE_URL = (
    "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true"
)

# Create '/data' directory if it doesn't exist
# data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
# os.makedirs(data_dir, exist_ok=True)
# Load datasets and save to '/data'

def create_id(*parts) -> str:
    """
    Create a deterministic ID by hashing input parts.

    Returns
    -------
    str
        Hex MD5 string used as a stable unique identifier.
    """
    md5 = hashlib.md5()
    md5.update("|".join("" if p is None else str(p) for p in parts).encode("utf-8"))
    return md5.hexdigest()


def stream_movies(source):
    """
    Yield JSON objects from a JSON Lines source (URL or local file).

    Parameters
    ----------
    source : str or Path
        Remote URL (http/https) or local file path.

    Yields
    ------
    dict
        One parsed JSON object per line.
    """
    src = str(source)
    is_url = src.lower().startswith(("http://", "https://"))
    if is_url:
        with urlopen(src) as resp:  # stdlib-only networking
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def etl(source: str = SOURCE_URL):
    """
    Run ETL and write normalized CSVs to ./data per Data Standards.

    Parameters
    ----------
    source : str
        JSON Lines URL or local file path.

    Side Effects
    ------------
    Writes five CSVs to ./data:
      - movies.csv (movie_uid, title, year, rating)
      - actors.csv (actor_id, actor_name)
      - movie_actors.csv (id, movie_uid, actor_id)
      - genres.csv (genre_id, genre)
      - movie_genres.csv (id, movie_uid, genre_id)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    movies_rows = []
    actors_map = {}
    movie_actors_rows = []
    genres_map = {}
    movie_genres_rows = []

    for rec in stream_movies(source):
        # Expected fields in this dataset
        title = rec.get("title")
        year = rec.get("year")
        rating = rec.get("rating")
        imdb_id = rec.get("imdb_id")

        movie_uid = imdb_id or create_id(title, year)
        movies_rows.append({"movie_uid": movie_uid, "title": title, "year": year, "rating": rating})

        # Genres: list of strings
        for g in rec.get("genres", []) or []:
            gid = create_id(g)
            genres_map[gid] = str(g)
            movie_genres_rows.append({
                "id": create_id(movie_uid, g),
                "movie_uid": movie_uid,
                "genre_id": gid
            })

        # Actors: list like [[actor_id, actor_name], ...]
        for pair in rec.get("actors", []) or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            actor_id, actor_name = pair[0], pair[1]
            if actor_id and actor_name:
                actors_map[actor_id] = actor_name
                movie_actors_rows.append({
                    "id": create_id(movie_uid, actor_id),
                    "movie_uid": movie_uid,
                    "actor_id": actor_id
                })

    # DataFrames (ensure unique IDs)
    movies_df = pd.DataFrame(movies_rows).drop_duplicates(subset=["movie_uid"])
    actors_df = pd.DataFrame(
        [{"actor_id": k, "actor_name": v} for k, v in actors_map.items()]
    ).drop_duplicates(subset=["actor_id"])
    movie_actors_df = pd.DataFrame(movie_actors_rows).drop_duplicates(subset=["id"])
    genres_df = pd.DataFrame(
        [{"genre_id": gid, "genre": name} for gid, name in genres_map.items()]
    ).drop_duplicates(subset=["genre_id"])
    movie_genres_df = pd.DataFrame(movie_genres_rows).drop_duplicates(subset=["id"])

    # Save to /data
    movies_df.to_csv(DATA_DIR / "movies.csv", index=False)
    actors_df.to_csv(DATA_DIR / "actors.csv", index=False)
    movie_actors_df.to_csv(DATA_DIR / "movie_actors.csv", index=False)
    genres_df.to_csv(DATA_DIR / "genres.csv", index=False)
    movie_genres_df.to_csv(DATA_DIR / "movie_genres.csv", index=False)

    print(
        f"ETL complete. Saved {len(movies_df)} movies, {len(actors_df)} actors, "
        f"{len(movie_actors_df)} castings, {len(genres_df)} genres, {len(movie_genres_df)} movie-genre links "
        f"to {DATA_DIR}"
    )


if __name__ == "__main__":
    etl()