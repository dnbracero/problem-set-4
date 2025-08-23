'''
You will run this problem set from main.py, so set things up accordingly
'''

from src.part1_etl import etl
from src.part2_network_centrality import nc
from src.part3_similar_actors_genre import sag

# Call functions / instanciate objects from the .py files
def main():
    # PART 1: Instanciate etl, saving the dataset in `./data/`
    etl()

    # PART 2: Call functions/instanciate objects for the network centrality analysis
    nc()
    print('NC complete.')

    # PART 3: Call functions/instanciate objects for similar actors by genre
    sag()
    print('SAG complete.')

if __name__ == "__main__":
    main()