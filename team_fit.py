import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

TEAMS = ['76ers', 'Bucks', 'Bulls', 'Cavaliers', 'Celtics', 'Clippers', 'Grizzlies', 'Hawks', 'Heat', 'Hornets', 'Jazz', 'Kings', 'Knicks', 'Lakers', 'Magic', 'Mavericks', 'Nets', 'Nuggets', 'Pacers', 'Pelicans', 'Pistons', 'Raptors', 'Rockets', 'Spurs', 'Suns', 'Thunder', 'Timberwolves', 'TrailBlazers', 'Warriors', 'Wizards']
TEAM_CODES = ['PHI', 'MIL', 'CHI', 'CLE', 'BOS', 'LAC', 'MEM', 'ATL', 'MIA', 'CHO', 'UTA', 'SAC', 'NYK', 'LAL', 'ORL', 'DAL', 'BRK', 'DEN', 'IND', 'NOP', 'DET', 'TOR', 'HOU', 'SAS', 'PHO', 'OKC', 'MIN', 'POR', 'GSW', 'WAS']
codes_dict = {TEAMS[i]: TEAM_CODES[i] for i in range(len(TEAM_CODES))}
YEARS = ['2018']#['2014', '2015', '2016', '2017', '2018']

teams_dict = {}
draft_data = pd.read_csv('2018_draft.csv').set_index('Tm')

for team in TEAMS:
	for year in YEARS:
		roster = pd.read_csv('team_info/' + team + 'Roster' + year + '.csv')
		advanced = pd.read_csv('team_info/' + team + 'Advanced' + year + '.csv')
		advanced['Player'] = advanced['Unnamed: 1']
		advanced.drop('Unnamed: 1', axis=1, inplace=True)
		joined = roster.set_index('Player').join(advanced.set_index('Player'))
		joined = joined[['Pos', 'WS']]
		joined = pd.get_dummies(joined['Pos']).join(joined).drop('Pos', axis=1)
		# get the position of the player they actually drafted first
		team_code = codes_dict[team]
		if (team_code in draft_data.index):
			position = draft_data.loc[team_code].iloc[0].loc['Pos'] if (len(draft_data.loc[team_code].shape) > 1) else draft_data.loc[team_code].loc['Pos']
		else:
			position = None
		teams_dict[team + '_' + year] = [joined.sort_values(by=['WS'], ascending=False), position]
 
## Want to predict which position a team picks given their positions and win shares at the positions
## then combine that with player skill to re-rank and pick
inputs = [v[0].to_numpy()[:12].flatten() for k, v in teams_dict.items() if v[1] is not None]
outputs = [v[1] for k, v in teams_dict.items() if v[1] is not None]
enc = OneHotEncoder()
model_target = enc.fit_transform(outputs)
print(enc.categories_)
classifier = MLPClassifier()
classifier.fit(inputs, model_target)

#classifier.predict_proba(test_set)
