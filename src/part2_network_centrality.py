'''
PART 2: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Build a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is inline with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import json
from itertools import combinations
from pathlib import Path

DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def nc() -> Path:
    """
    Build a co-actor network graph from the dataset in `data/`, compute degree centrality,
    print summary information, and export an edge list CSV including the required columns.

    Returns
    -------
    Path
        Absolute path to the exported CSV in `data/`.
    """    
    # Build the graph
    g = nx.Graph()

    # Required inputs from Part 1
    actors_csv = DATA_DIR / "actors.csv"
    movie_actors_csv = DATA_DIR / "movie_actors.csv"

    if not actors_csv.exists() or not movie_actors_csv.exists():
        raise FileNotFoundError(
            "Expected './data/actors.csv' and './data/movie_actors.csv'. "
            "Please run Part 1 ETL first."
        )

    # Read normalized tables
    actors_df = pd.read_csv(actors_csv, dtype=str)
    movie_actors_df = pd.read_csv(movie_actors_csv, dtype=str)

    # Validate schema
    if not {"actor_id", "actor_name"}.issubset(actors_df.columns):
        raise ValueError("actors.csv must include columns: {'actor_id','actor_name'}")
    if not {"movie_uid", "actor_id"}.issubset(movie_actors_df.columns):
        raise ValueError("movie_actors.csv must include columns: {'movie_uid','actor_id'}")

    # Map actor_id -> actor_name and add nodes
    actor_name_map = dict(zip(actors_df["actor_id"], actors_df["actor_name"]))
    for aid, aname in actor_name_map.items():
        g.add_node(str(aid), name=str(aname))

    # Build weighted co-appearance edges by movie
    edge_weights: dict[tuple[str, str], int] = {}
    for movie_uid, grp in movie_actors_df.groupby("movie_uid", sort=False):
        actor_ids = sorted(set(grp["actor_id"].dropna().astype(str)))
        for a, b in combinations(actor_ids, 2):
            key = (a, b) if a <= b else (b, a)
            edge_weights[key] = edge_weights.get(key, 0) + 1

    for (a, b), wt in edge_weights.items():
        g.add_edge(a, b, weight=int(wt))

    # Print the info below
    print("Nodes:", len(g.nodes))

    # Print the 10 the most central nodes (degree centrality)
    deg_cent = nx.degree_centrality(g)
    top10 = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 by degree centrality (actor_id, actor_name, centrality):")
    for aid, score in top10:
        print(f"  {aid:>12}  {g.nodes[aid].get('name', '')}  {score:.5f}")

    # Build the required edge list dataframe
    rows = []
    for u, v, data in g.edges(data=True):
        # Deterministic left/right ordering by actor_id
        left_id, right_id = (u, v) if u <= v else (v, u)
        left_name = g.nodes[left_id].get("name", "")
        right_name = g.nodes[right_id].get("name", "")
        weight = int(data.get("weight", 1))
        edge_uid = f"{left_id}__{right_id}"

        rows.append(
            {
                "uid": edge_uid,                 # unique row id (Data Standards)
                "left_actor_id": left_id,
                "left_actor_name": left_name,
                "<->": "<->",
                "right_actor_id": right_id,
                "right_actor_name": right_name,
                "weight": weight,
            }
        )

    df_edges = pd.DataFrame(rows)

    # Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_csv = DATA_DIR / f"network_centrality_{timestamp}.csv"
    df_edges.to_csv(out_csv, index=False)
    print(f"Exported edge list with weights to: {out_csv}")

    return out_csv



if __name__ == "__main__":
    nc()

