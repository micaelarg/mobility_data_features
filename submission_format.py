import pandas as pd

test_df = pd.read_csv('/home/prisma/Documents/GitHub/mobility_data_features/data/test.csv')
predictions_df = pd.read_csv('/home/prisma/Documents/GitHub/mobility_data_features/results/predictions.csv')

missing_hex_ids = set(test_df['hex_id']) - set(predictions_df['hex_id'])
missing_df = pd.DataFrame({'hex_id': list(missing_hex_ids), 'cost_of_living': [None] * len(missing_hex_ids)})
missing_df['cost_of_living'] = pd.to_numeric(missing_df['cost_of_living'], errors='coerce').fillna(0)

complete_predictions = pd.concat([predictions_df, missing_df], ignore_index=True)
complete_predictions.to_csv('/home/prisma/Documents/GitHub/mobility_data_features/submission.csv', index=False)


# comment
# Clear improvements to the model and submission can be found here: https://github.com/micaelarg/mobility_data_features/blob/main/improvements.txt