import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go


model = 'inception_v3'
df = pd.read_csv('results.csv')
df.set_index(df.columns[0], inplace=True)
df = df.loc[df.index.str.endswith('_avg')]
df.index.names = ['different eps']
print(df)

df_fgsm = df.loc[:, df.columns.isin(['FGSM_TOP5_'+model, 'FGSM_TOP5_TWO_STREAM',
                                     # 'FGSM_TOP5_'+model, 'FGSM_TOP5_TWO_STREAM',
                                     # 'FGSM_TOP10_'+model, 'FGSM_TOP10_TWO_STREAM',
                                     # 'FGSM_fscore_'+model, 'FGSM_fscore_TWO_STREAM',
                                     # 'FGSM_precisions_'+model, 'FGSM_precisions_TWO_STREAM',
                                     # 'FGSM_recalls_'+model, 'FGSM_recalls_TWO_STREAM',
                                         
                                         ])]
df_pgd = df.loc[:, df.columns.isin([  'PGD_TOP5_'+model, 'PGD_TOP5_TWO_STREAM', 
	# 'PGD_TOP5_'+model, 'PGD_TOP5_TWO_STREAM',
	#  'PGD_TOP10_'+model, 'PGD_TOP10_TWO_STREAM',
	#   'PGD_fscore_'+model, 'PGD_fscore_TWO_STREAM',
	#    'PGD_precisions_'+model, 'PGD_precisions_TWO_STREAM',
	#      'PGD_recalls_'+model, 'PGD_recalls_TWO_STREAM' 


])]

# print(df_fgsm)
fig = px.line(df_fgsm, title='Comparison of FGSM attack on Inception and Ours network (Two-stream)')
fig.update_layout(
    yaxis_title="ACCURACY",
)
fig.write_image("results_inception_fgsm_top5.png")

# print(df_pgd)
fig = px.line(df_pgd, title='Comparison of PGD attack on Iinception and Ours network (Two-stream)')
fig.update_layout(
    yaxis_title="ACCURACY",
)
fig.write_image("results_inception_pgd_top5.png")