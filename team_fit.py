import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

TEAMS = ['76ers', 'Bucks', 'Bulls', 'Cavaliers', 'Celtics', 'Clippers', 'Grizzlies', 'Hawks', 'Heat', 'Hornets', 'Jazz', 'Kings', 'Knicks', 'Lakers', 'Magic', 'Mavericks', 'Nets', 'Nuggets', 'Pacers', 'Pelicans', 'Pistons', 'Raptors', 'Rockets', 'Spurs', 'Suns', 'Thunder', 'Timberwolves', 'TrailBlazers', 'Warriors', 'Wizards']
TEAM_CODES = ['PHI', 'MIL', 'CHI', 'CLE', 'BOS', 'LAC', 'MEM', 'ATL', 'MIA', 'CHO', 'UTA', 'SAC', 'NYK', 'LAL', 'ORL', 'DAL', 'BRK', 'DEN', 'IND', 'NOP', 'DET', 'TOR', 'HOU', 'SAS', 'PHO', 'OKC', 'MIN', 'POR', 'GSW', 'WAS']
codes_dict = {TEAMS[i]: TEAM_CODES[i] for i in range(len(TEAM_CODES))}
YEARS = ['2015', '2016', '2017', '2018'] #['2014', '2015', '2016', '2017', '2018']

teams_dict = {}
draft_data = {year: pd.read_csv(year + '_draft.csv', sep='\t').set_index('Tm') for year in YEARS[:3]}
draft_data['2018'] = pd.read_csv('2018_draft.csv').set_index('Tm') # different format for the most recent draft

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
		if (team_code in draft_data[year].index):
			position = draft_data[year].loc[team_code].iloc[0].loc['Pos'] if (len(draft_data[year].loc[team_code].shape) > 1) else draft_data[year].loc[team_code].loc['Pos']
		else:
			position = None
		teams_dict[team + '_' + year] = [joined.sort_values(by=['WS'], ascending=False), position]
 
# Make train and test data sets
train_inputs = [v[0].to_numpy()[:12].flatten() for k, v in teams_dict.items() if v[1] is not None and '2018' not in k]
test_inputs = [v[0].to_numpy()[:12].flatten() for k, v in teams_dict.items() if v[1] is not None and '2018' in k]
train_outputs = [v[1] for k, v in teams_dict.items() if v[1] is not None and '2018' not in k]
test_outputs = [v[1] for k, v in teams_dict.items() if v[1] is not None and '2018' in k]

# preprocess outputs (positions) to be one-hot vectors
enc = OneHotEncoder()
train_target = enc.fit_transform(np.array(train_outputs).reshape(-1, 1))
test_target = enc.transform(np.array(test_outputs).reshape(-1, 1))
print(enc.categories_)

# instantiate and run the classifier
classifier = MLPClassifier(max_iter=5000)
classifier.fit(train_inputs, train_target)
predicted = [enc.categories_[0][np.argmax(a)] for a in classifier.predict_proba(test_inputs)]
print(classifier.score(test_inputs, test_target))

correct = 0
for i in range(len(predicted)):
	print(predicted[i], test_outputs[i])
	if predicted[i] == test_outputs[i]: 
		correct += 1
acc = correct / len(predicted)
print('Accuracy = ' + str(acc))

