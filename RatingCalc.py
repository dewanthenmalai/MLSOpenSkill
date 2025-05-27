import openskill
import polars as pl
from openskill.models import PlackettLuce

# Data loaded from https://fixturedownload.com/results/mls-2025pl.String, "Away": pl.String, "Home Win": pl.FLoat64, "Draw": pl.FLoat64, "Away Win": pl.FLoat64}
def parse_match(datarow):
    scores = datarow["Result"].split()
    return { datarow["Home Team"]: int(scores[0]), datarow["Away Team"]: int(scores[2]) }

teamNames = [ 'SEA', 'DAL', 'SD', 'LA', 'CLT', 'CLB', 'DC', 'RBNY', 'MTL', 'LAFC', 'ORL', 'POR', 'PHI', 'MIA', 'TOR', 'NSH', 'SKC', 'NE', 'MIN', 'ATX', 'COL', 'STL', 'RSL', 'VAN', 'SJ', 'HOU', 'NYC', 'CHI', 'ATL', 'CIN']

model = PlackettLuce()

teams = {n: model.rating(name=n) for n in teamNames}

df_results = pl.read_csv("mls-2025.csv")
played_matches = df_results.filter(pl.col("Result").is_not_null()).select("Round Number", "Home Team", "Away Team", "Result")

matchday = max(played_matches.get_column("Round Number"))

for i in range(1, matchday+1):
    current__matchday = played_matches.filter(pl.col("Round Number") == i)
    for match in current__matchday.iter_rows(named=True):
        parsed = parse_match(match)
        game = [[teams[match["Home Team"]]], [teams[match["Away Team"]]]]
        score = [parsed[match["Home Team"]], parsed[match["Home Team"]]]
        [[home], [away]] = model.rate(game, scores=score)
        teams[match["Home Team"]] = home
        teams[match["Away Team"]] = away

# Predict upcoming match results
upcoming_matches = df_results.filter(pl.col("Round Number").eq(matchday+1)).select("Home Team", "Away Team")

df_predict = pl.DataFrame([], schema={"Home": pl.String, "Away": pl.String, "Home Win": pl.Float64, "Draw": pl.Float64, "Away Win": pl.Float64})

for m in upcoming_matches.iter_rows(named=True):
    home_win, away_win = model.predict_win([[teams[m["Home Team"]]], [teams[m["Away Team"]]]])
    draw = model.predict_draw([[teams[m["Home Team"]]], [teams[m["Away Team"]]]])
    new_row = pl.DataFrame({"Home": m["Home Team"], "Away": m["Away Team"], "Home Win": home_win, "Draw": draw, "Away Win": away_win})
    df_predict = pl.concat([df_predict, new_row])

# create ranking
league = [[t] for t in teams.values()]
rank_predictions = model.predict_rank(league)
team_rankings = [(n, r[0], r[1]) for n, r in zip(teamNames, rank_predictions)]

team_rankings.sort(key=lambda r: r[1])

df_rankings = pl.DataFrame(team_rankings, schema=["Team", "Rank", "Probability"], orient="row")

print(df_rankings)
print(df_predict)