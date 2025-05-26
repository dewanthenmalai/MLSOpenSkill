import openskill
import polars as pl
from openskill.models import PlackettLuce

# Data loaded from https://fixturedownload.com/results/mls-2025

def parse_match(datarow):
    scores = datarow["Result"].split()
    return { datarow["Home Team"]: int(scores[0]), datarow["Away Team"]: int(scores[2]) }

teamNames = [ 'SEA', 'DAL', 'SD', 'LA', 'CLT', 'CLB', 'DC', 'RBNY', 'MTL', 'LAFC', 'ORL', 'POR', 'PHI', 'MIA', 'TOR', 'NSH', 'SKC', 'NE', 'MIN', 'ATX', 'COL', 'STL', 'RSL', 'VAN', 'SJ', 'HOU', 'NYC', 'CHI', 'ATL', 'CIN']

model = PlackettLuce()

teams = {n: model.rating(name=n) for n in teamNames}

df_results = pl.read_csv("mls-2025.csv")
played_matches = df_results.filter(pl.col("Result").is_not_null()).select("Round Number", "Home Team", "Away Team", "Result")

for i in range(1, max(played_matches.get_column("Round Number"))+1):
    matchday = played_matches.filter(pl.col("Round Number") == i)
    for match in matchday.iter_rows(named=True):
        parsed = parse_match(match)
        game = [[teams[match["Home Team"]]], [teams[match["Away Team"]]]]
        score = [parsed[match["Home Team"]], parsed[match["Home Team"]]]
        [[home], [away]] = model.rate(game, scores=score)
        teams[match["Home Team"]] = home
        teams[match["Away Team"]] = away

league = [[t] for t in teams.values()]
rank_predictions = model.predict_rank(league)

for name, rank in zip(teamNames, rank_predictions):
    print(f"Team: {name}, Rank: {rank[0]}, Probability: {rank[1]}")