import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score

##################### CONSTANTS ######################
# the name of the file with the college (feature) data
COLLEGE_DATA_FILE = 'last_four_seasons.csv'

# the info for the file with the pro (target) data
DRAFT_SUFFIX = '_draft.csv'
DRAFT_YEARS = ['2014', '2015', '2016', '2017', '2018']

# info for the position picking (team needs) classifier
TEAMS = ['76ers', 'Bucks', 'Bulls', 'Cavaliers', 'Celtics', 'Clippers', 'Grizzlies', 'Hawks', 'Heat', 'Hornets', 'Jazz', 'Kings', 'Knicks', 'Lakers', 'Magic', 'Mavericks', 'Nets', 'Nuggets', 'Pacers', 'Pelicans', 'Pistons', 'Raptors', 'Rockets', 'Spurs', 'Suns', 'Thunder', 'Timberwolves', 'TrailBlazers', 'Warriors', 'Wizards']
TEAM_CODES = ['PHI', 'MIL', 'CHI', 'CLE', 'BOS', 'LAC', 'MEM', 'ATL', 'MIA', 'CHO', 'UTA', 'SAC', 'NYK', 'LAL', 'ORL', 'DAL', 'BRK', 'DEN', 'IND', 'NOP', 'DET', 'TOR', 'HOU', 'SAS', 'PHO', 'OKC', 'MIN', 'POR', 'GSW', 'WAS']
codes_dict = {TEAMS[i]: TEAM_CODES[i] for i in range(len(TEAM_CODES))}
YEARS = ['2014', '2015', '2016', '2017', '2018']

############# BEGINNING OF PLAYER MODEL ################
# read in data
college_data = pd.read_csv(COLLEGE_DATA_FILE)
player_draft_data_test = pd.read_csv(DRAFT_YEARS[0] + DRAFT_SUFFIX)
player_draft_data = pd.read_csv(DRAFT_YEARS[1] + DRAFT_SUFFIX, sep='\t')
for i in range(2, len(DRAFT_YEARS)):
	if (i == len(DRAFT_YEARS) - 1):
		next_year = pd.read_csv(DRAFT_YEARS[i] + DRAFT_SUFFIX)
	else:
		next_year = pd.read_csv(DRAFT_YEARS[i] + DRAFT_SUFFIX, sep='\t')
	player_draft_data = pd.concat([player_draft_data, next_year])

# get rid of messy IDs in player names for the join
college_data['Player'] = college_data['Player'].apply(lambda n: n[:n.find('\\')])
print(player_draft_data['Player'])
player_draft_data['Player'] = player_draft_data['Player'].apply(lambda n: n[:n.find('\\')])
player_draft_data_test['Player'] = player_draft_data_test['Player'].apply(lambda n: n[:n.find('\\')])

# get pro info
train_joined = player_draft_data.set_index('Player').join(college_data.set_index('Player'), lsuffix='_pro', rsuffix='_coll')
test_joined = player_draft_data_test.set_index('Player').join(college_data.set_index('Player'), lsuffix='_pro', rsuffix='_coll')

# then drop unnecessary columns
drop_cols = ['Rk_coll', 'Pk', 'Tm', 'College', 'Yrs', 'PTS',
       'TRB', 'AST', 'FG%', '3P%', 'FT%', 'MP.1', 'PTS.1', 'TRB.1', 'AST.1', 'WS/48', 'VORP', 'Rk_pro', 'From', 'To',
       'School', 'Conf', 'G_pro', 'MP_pro', 'BPM_pro']
train_df = train_joined.drop(drop_cols, axis=1).dropna()
test_df = test_joined.drop(drop_cols, axis=1).dropna()

# divide into feature and target (X and Y) arrays
train_target = train_df['WS_pro'].to_numpy(dtype=float)
test_target = test_df['WS_pro'].to_numpy(dtype=float)
train_pos, test_pos = train_df['Pos'], test_df['Pos']
train_features = train_df.drop(['WS_pro', 'Pos'], axis=1).to_numpy(dtype=float)
test_features = test_df.drop(['WS_pro', 'Pos'], axis=1).to_numpy(dtype=float)

# try a linear regressor, see how bad it is?
regressor = LinearRegression().fit(train_features, train_target)
predictions = regressor.predict(test_features)
predicted_ranking = [[test_df.index[i], predictions[i], test_pos.iloc[i]] for i in range(len(predictions))]
predicted_ranking.sort(key=lambda l: l[1], reverse=True)
#r2 = r2_score(test_target, predictions)
#print(r2)

########### END OF PLAYER MODEL ###############

########## BEGINNING OF TEAM MODEL ############
teams_dict = {}
draft_data = {}
for year in DRAFT_YEARS[1:]:
	if year != '2018':
		draft_data[year] = pd.read_csv(year + '_draft.csv', sep='\t').set_index('Tm')
	else:
		draft_data[year] = pd.read_csv(year + '_draft.csv').set_index('Tm')

draft_data['2014'] = pd.read_csv('2014_draft.csv').set_index('Tm') # different format for the most recent draft

for t in TEAMS:
	for year in YEARS:
		if (t == 'Hornets' and year == '2014'):
			team = 'Bobcats'
		else:
			team = t
		roster = pd.read_csv('team_info/' + team + 'Roster' + year + '.csv')
		advanced = pd.read_csv('team_info/' + team + 'Advanced' + year + '.csv')
		advanced['Player'] = advanced['Unnamed: 1']
		advanced.drop('Unnamed: 1', axis=1, inplace=True)
		joined = roster.set_index('Player').join(advanced.set_index('Player'))
		joined = joined[['Pos', 'WS']]
		joined = pd.get_dummies(joined['Pos']).join(joined).drop('Pos', axis=1)
		# get the position of the player they actually drafted first
		team_code = codes_dict[team] if team != 'Bobcats' else 'CHH'
		if (team_code in draft_data[year].index):
			position = draft_data[year].loc[team_code].iloc[0].loc['Pos'] if (len(draft_data[year].loc[team_code].shape) > 1) else draft_data[year].loc[team_code].loc['Pos']
		else:
			position = None
		teams_dict[team + '_' + year] = [joined.sort_values(by=['WS'], ascending=False), position]
 
# Make train and test data sets
train_inputs = [v[0].to_numpy()[:12].flatten() for k, v in teams_dict.items() if v[1] is not None and '2014' not in k]
test_inputs = [v[0].to_numpy()[:12].flatten() for k, v in teams_dict.items() if v[1] is not None and '2014' in k]
train_outputs = [v[1] for k, v in teams_dict.items() if v[1] is not None and '2014' not in k]
test_outputs = [v[1] for k, v in teams_dict.items() if v[1] is not None and '2014' in k]

# preprocess outputs (positions) to be one-hot vectors
enc = OneHotEncoder()
train_target = enc.fit_transform(np.array(train_outputs).reshape(-1, 1))
test_target = enc.transform(np.array(test_outputs).reshape(-1, 1))

# instantiate and run the classifier
classifier = MLPClassifier(max_iter=5000)
classifier.fit(train_inputs, train_target)
predicted_position = {draft_data['2014'].index.unique()[i]: classifier.predict_proba(test_inputs)[i] for i in range(len(draft_data['2014'].index.unique()))}
#predicted = [enc.categories_[0][np.argmax(a)] for a in classifier.predict_proba(test_inputs)]
#print(classifier.score(test_inputs, test_target))

taken = set()
predicted_draft = []
for t in draft_data['2014'].index:
	pos_ranks = {enc.categories_[0][i]: predicted_position[t][i] for i in range(len(enc.categories_[0]))}
	ranking_copy = copy.deepcopy(predicted_ranking)
	for player in ranking_copy:
		player.append(pos_ranks[player[2]])
		player.append(player[1] * player[3])
	ranking_copy.sort(key=lambda k: k[4], reverse=True)
	for prospect in ranking_copy:
		if prospect[0] not in taken:
			predicted_draft.append([t, prospect[0], prospect[2], prospect[4]])
			taken.add(prospect[0])
			break

for i, p in enumerate(predicted_draft):
	print(str(i + 1) + '\t' + p[1] + '\t' + str(p[3]))

'''
Predicted Pick | Actual Pick | Error | Player | Predicted WS
1 	6 	5	Mohamed Bamba 		15.938112209653504
2 	1 	1	Deandre Ayton 		9.063310218949397
3 	50 	47	Alize Johnson 		8.036329099291407
4 	27 	23	Robert Williams 	7.344430393508716 ??
5 	41 	36	Jarred Vanderbilt 	7.331831981862763
6 	42 	36	Bruce Brown 		5.704142724759819
7 	16 	9	Zhaire Smith 		5.619596232765488
8 	32 	24	Jevon Carter 		4.949891309805992
9 	30 	21	Omari Spellman 		4.880486853311766
10 	28 	18	Jacob Evans 		4.765113024944
11 	54 	43	Shake Milton 		4.301225280870909
12 	48 	36	Keita Bates-Diop 	4.098643435702122
13 	21 	8	Grayson Allen 		3.687432999346136
14 	26 	12	Landry Shamet 		3.3515534435207144
15 	10 	5	Mikal Bridges 		3.2271549083432447
16 	11 	5	Shai Gilgeous-Alexander 2.9938506353578447
17 	33 	16	Jalen Brunson 		2.5137256177497207
18 	25 	7	Moritz Wagner 		2.4955546288054435
19 	12 	7	Miles Bridges 		2.464818310422899
20 	46 	26	De'Anthony Melton 	2.3749201896231327
21 	49 	28	Chimezie Metu 		2.1867571200878544
22 	8 	14	Collin Sexton 		1.9774950493195291
23 	5 	18	Trae Young 			1.8182253912931188
24 	34 	10	Devonte' Graham 	1.7913636424043489
25 	20 	5	Josh Okogie 		1.4083053514323183
26 	27 	1	Robert Williams 	1.2124193758782624
27 	9 	18	Kevin Knox 			1.0863297489297992
28 	17 	11	Donte DiVincenzo 	0.9995646773151279
29 	52 	23	Vince Edwards 		0.8840297504223926
30 	58 	28	Thomas Welsh 		0.8674879228116623
31 	19 	12	Kevin Huerter 		0.584780291686144
32 	15 	17	Troy Brown 			0.2368034658791558
33 	22 	11	Chandler Hutchison 	-0.06904866735312964
34 	59 	25	George King 		-0.33539948624076743
35 	18 	17	Lonnie Walker 		-0.5963813589628799
36 	38 	2	Khyri Thomas 		-0.9676013865003767
37 	45 	8	Hamidou Diallo 		-2.3868393973071846
38 	13 	25	Jerome Robinson 	-2.7413891603670777
39 	35 	4	Melvin Frazier 		-2.742614189612013
40 	47 	7	Sviatoslav Mykhailiuk -3.1285094260110213
41 	23 	18	Aaron Holiday 		-3.4196988108696615
42 	60 	18	Kostas Antetokounmpo -7.7000823712347
'''
