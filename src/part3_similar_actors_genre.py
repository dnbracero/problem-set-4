'''
PART 2: SIMILAR ACTROS BY GENRE

Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

#Write your code below

import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances

from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_normalized_tables(base_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the normalized CSVs produced in Part 1.

    Parameters
    ----------
    base_dir : Path
        Directory containing actors.csv, movie_actors.csv, genres.csv, movie_genres.csv.

    Returns
    -------
    (actors_df, movie_actors_df, genres_df, movie_genres_df) : tuple[pd.DataFrame, ...]
        DataFrames with all columns as strings.

    Raises
    ------
    FileNotFoundError
        If any of the required CSVs is missing.
    ValueError
        If required columns are missing from any table.
    """
    actors_csv = base_dir / "actors.csv"
    movie_actors_csv = base_dir / "movie_actors.csv"
    genres_csv = base_dir / "genres.csv"
    movie_genres_csv = base_dir / "movie_genres.csv"

    missing = [p.name for p in [actors_csv, movie_actors_csv, genres_csv, movie_genres_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {base_dir}: {', '.join(missing)}")

    actors_df = pd.read_csv(actors_csv, dtype=str)
    movie_actors_df = pd.read_csv(movie_actors_csv, dtype=str)
    genres_df = pd.read_csv(genres_csv, dtype=str)
    movie_genres_df = pd.read_csv(movie_genres_csv, dtype=str)

    if not {"actor_id", "actor_name"}.issubset(actors_df.columns):
        raise ValueError("actors.csv must include columns: {'actor_id','actor_name'}")
    if not {"movie_uid", "actor_id"}.issubset(movie_actors_df.columns):
        raise ValueError("movie_actors.csv must include columns: {'movie_uid','actor_id'}")
    if not {"genre_id", "genre"}.issubset(genres_df.columns):
        raise ValueError("genres.csv must include columns: {'genre_id','genre'}")
    if not {"movie_uid", "genre_id"}.issubset(movie_genres_df.columns):
        raise ValueError("movie_genres.csv must include columns: {'movie_uid','genre_id'}")

    return actors_df, movie_actors_df, genres_df, movie_genres_df


def actor_genre_matrix(
    actors_df: pd.DataFrame,
    movie_actors_df: pd.DataFrame,
    genres_df: pd.DataFrame,
    movie_genres_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Build an actor-by-genre count matrix.

    Parameters
    ----------
    actors_df : pd.DataFrame
        Columns: actor_id, actor_name
    movie_actors_df : pd.DataFrame
        Columns: movie_uid, actor_id
    genres_df : pd.DataFrame
        Columns: genre_id, genre
    movie_genres_df : pd.DataFrame
        Columns: movie_uid, genre_id

    Returns
    -------
    (df_counts, actor_names) : tuple
        df_counts: DataFrame with index=actor_id, columns=genre, values=count
        actor_names: dict mapping actor_id -> actor_name
    """
    # Join to get (movie_uid, actor_id, genre_id)
    ma_mg = movie_actors_df.merge(movie_genres_df, on="movie_uid", how="inner")
    # Bring in genre label
    ma_mg = ma_mg.merge(genres_df, on="genre_id", how="left")

    # Count appearances by (actor_id, genre)
    counts = (
        ma_mg.groupby(["actor_id", "genre"], dropna=True)
        .size()
        .reset_index(name="count")
    )

    # Pivot to actor-by-genre matrix
    df_counts = counts.pivot(index="actor_id", columns="genre", values="count").fillna(0).astype(np.int32)
    # Stable column order (alphabetical)
    df_counts = df_counts.reindex(sorted(df_counts.columns), axis=1)

    # Actor name map
    actor_names = dict(zip(actors_df["actor_id"], actors_df["actor_name"]))

    # Keep only actors present in actors_df to maintain consistency
    df_counts = df_counts.reindex(actors_df["actor_id"].tolist(), fill_value=0)

    return df_counts, actor_names


def sag(
    query_actor_id: str = "nm1165110",
    top_k: int = 10,
    base_dir: Path = DATA_DIR,
) -> Path:
    """
    Similar (genre) actors: compute top-K nearest neighbors by cosine distance and
    compare with Euclidean distance.

    Parameters
    ----------
    query_actor_id : str
        Actor ID to use as the query (e.g., 'nm1165110' for Chris Hemsworth).
    top_k : int
        Number of nearest neighbors to report.
    base_dir : Path
        Directory with the normalized CSVs (./data).

    Returns
    -------
    Path
        Path to the written CSV file under ./data (timestamped).
    """
    actors_df, movie_actors_df, genres_df, movie_genres_df = load_normalized_tables(base_dir)
    df_counts, actor_names = actor_genre_matrix(actors_df, movie_actors_df, genres_df, movie_genres_df)

    if query_actor_id not in df_counts.index:
        raise ValueError(
            f"Query actor_id '{query_actor_id}' not found. "
            f"Matrix has {df_counts.shape[0]} actors."
        )

    X = df_counts.to_numpy(dtype=np.float64)
    actor_ids = df_counts.index.to_list()
    idx = {aid: i for i, aid in enumerate(actor_ids)}
    q_idx = idx[query_actor_id]
    q_vec = X[q_idx : q_idx + 1, :]  # 1xG row

    # Cosine distance (pairwise API supports 'cosine' directly)
    cos_dist = cosine_distances(q_vec, X).ravel()
    # Euclidean distance via DistanceMetric per instructions
    euc = DistanceMetric.get_metric("euclidean")
    euc_dist = euc.pairwise(q_vec, X).ravel()

    # Handle any NaNs (e.g., zero vectors in cosine) conservatively
    cos_dist = np.nan_to_num(cos_dist, nan=1.0, posinf=1.0, neginf=1.0)

    # Exclude self
    mask = np.ones_like(cos_dist, dtype=bool)
    mask[q_idx] = False

    # Build candidate table
    cand = pd.DataFrame(
        {
            "actor_id": actor_ids,
            "actor_name": [actor_names.get(a, "") for a in actor_ids],
            "cosine_distance": cos_dist,
            "euclidean_distance": euc_dist,
        }
    )[mask]

    # Top-K by cosine distance (smaller is more similar)
    top_cos = cand.sort_values("cosine_distance", ascending=True).head(top_k).copy()
    top_cos.insert(0, "rank", range(1, len(top_cos) + 1))
    top_cos.insert(0, "uid", [f"sim_{query_actor_id}__{a}" for a in top_cos["actor_id"]])
    top_cos.insert(1, "query_actor_id", query_actor_id)
    top_cos.insert(2, "query_actor_name", actor_names.get(query_actor_id, ""))

    # Comparison list for Euclidean distance
    top_euc = cand.sort_values("euclidean_distance", ascending=True).head(top_k)
    cos_list = list(top_cos["actor_id"])
    euc_list = list(top_euc["actor_id"])
    only_cos = [a for a in cos_list if a not in euc_list]
    only_euc = [a for a in euc_list if a not in cos_list]

    print(
        f"Cosine vs Euclidean for '{query_actor_id}' ({actor_names.get(query_actor_id, '')}) — "
        f"only in cosine: {only_cos if only_cos else '—'}, only in euclidean: {only_euc if only_euc else '—'}"
    )

    # Write output CSV with timestamped filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_csv = base_dir / f"similar_actors_genre_{timestamp}.csv"
    cols = [
        "uid",
        "rank",
        "query_actor_id",
        "query_actor_name",
        "actor_id",
        "actor_name",
        "cosine_distance",
        "euclidean_distance",
    ]
    top_cos[cols].to_csv(out_csv, index=False)
    print(f"Exported top-{top_k} cosine-nearest actors to: {out_csv}")

    return out_csv


if __name__ == "__main__":
    sag()