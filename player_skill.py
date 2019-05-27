import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# the name of the file with the college (feature) data
COLLEGE_DATA_FILE = 'last_four_seasons.csv'

# the name of the file with the pro (target) data
DRAFT_DATA_FILE = 'last_four_drafts.csv'

# read in data
college_data = pd.read_csv(COLLEGE_DATA_FILE)
draft_data = pd.read_csv(DRAFT_DATA_FILE)

# get rid of messy IDs in player names for the join
college_data['Player'] = college_data['Player'].apply(lambda n: n[:n.find('\\')])
draft_data['Player'] = draft_data['Player'].apply(lambda n: n[:n.find('\\')])

# get pro info
train_joined = draft_data[60:].set_index('Player').join(college_data.set_index('Player'), lsuffix='_pro', rsuffix='_coll')
test_joined = draft_data[:60].set_index('Player').join(college_data.set_index('Player'), lsuffix='_pro', rsuffix='_coll')

# then drop unnecessary columns
drop_cols = ['Rk_coll', 'Pk', 'Tm', 'College', 'Yrs', 'PTS',
       'TRB', 'AST', 'FG%', '3P%', 'FT%', 'MP.1', 'PTS.1', 'TRB.1', 'AST.1', 'WS/48', 'VORP', 'Rk_pro', 'From', 'To',
       'School', 'Conf', 'G_pro', 'MP_pro', 'BPM_pro']
train_df = train_joined.drop(drop_cols, axis=1).dropna()
test_df = test_joined.drop(drop_cols, axis=1).dropna()

# divide into feature and target (X and Y) arrays
train_target = train_df['WS_pro'].to_numpy(dtype=float)
test_target = test_df['WS_pro'].to_numpy(dtype=float)
train_features = train_df.drop('WS_pro', axis=1).to_numpy(dtype=float)
test_features = test_df.drop('WS_pro', axis=1).to_numpy(dtype=float)

# try a linear regressor, see how bad it is?
regressor = LinearRegression().fit(train_features, train_target)
predictions = regressor.predict(test_features)
predicted_ranking = [[test_df.index[i], predictions[i]] for i in range(len(predictions))]
predicted_ranking.sort(key=lambda l: l[1], reverse=True)
#r2 = r2_score(test_target, predictions)
#print(r2)

for i, r in enumerate(predicted_ranking):
	print(i + 1, r[0], str(r[1]))

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
