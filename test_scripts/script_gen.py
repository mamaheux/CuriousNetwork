data_agm = [0, 1]
starting_feature_map=[2, 4, 8]
growth_factor=[2, 3]

for da in data_agm:
	for fm in starting_feature_map:
		for gf in growth_factor:
			call_str = 'python train.py --train_path test/path --learning_rate 0.001 --epoch_count 20'
			print(call_str)

