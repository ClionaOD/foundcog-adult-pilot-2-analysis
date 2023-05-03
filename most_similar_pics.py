import os
import pickle
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis_funcs import get_rois

subjlist = [f'{subjind:02}' for subjind in range(2,18)]
n_subj = len(subjlist)

pic_results_path = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/with-confounds/models'
#vid_results_path = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/video_betas'

vid_results_path = "/home/CUSACKLAB/clionaodoherty/foundcog_adult_pilot/models" #/sub-02/sub-02_task-video_remap-1_betas.pickle

rois = ['earlyvisual','ventralvisual','scene-occipital']
pic_conditions = [
                'seabird', 'crab', 'fish', 'seashell',
                'waiter', 'dishware', 'spoon', 'food',
                'tree_', 'flower_', 'rock', 'squirrel',
                'sink', 'shampoo', 'rubberduck', 'towel',
                'shelves', 'shoppingcart', 'soda', 'car',
                'dog', 'cat', 'ball', 'fence'
                ]
#vid_conditions = ['bathsong.mp4', 'dog.mp4', 'new_orleans.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'piper.mp4']
vid_conditions =['pingu.mp4', 'moana.mp4', 'brother_bear.mp4', 'bathsong.mp4', 'playground.mp4', 'rio_jungle_jig.mp4', 'bedtime.mp4', 'kids_kitchen.mp4', 'ratatouille.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'real_cars.mp4', 'new_orleans.mp4', 'dog.mp4', 'piper.mp4', 'our_planet.mp4']
pic_vid_mapping = { 
                    'bathsong.mp4':['sink', 'shampoo', 'rubberduck', 'towel'],
                    'minions_supermarket.mp4':['shelves', 'shoppingcart', 'soda', 'car'], 
                    'forest.mp4':['tree_', 'flower_', 'rock', 'squirrel'],
                    'new_orleans.mp4':['waiter', 'dishware', 'spoon', 'food'], 
                    'dog.mp4':['dog','cat', 'ball', 'fence'], 
                    'piper.mp4':['seabird','crab', 'fish', 'seashell'],
                    'moana.mp4':['seabird','crab', 'fish', 'seashell']}

n_pics = 24
n_vids = 16

#store mean across run responses for each subj individually
evc_pic_Bs = np.zeros((n_subj,1328,n_pics))
evc_vid_Bs = np.zeros((n_subj,1328,n_vids))

vvc_pic_Bs = np.zeros((n_subj,400,n_pics))
vvc_vid_Bs = np.zeros((n_subj,400,n_vids))

scene_pic_Bs = np.zeros((n_subj,601,n_pics))
scene_vid_Bs = np.zeros((n_subj,601,n_vids))

for idx, subj in enumerate(subjlist):
    pic_betas = pd.read_pickle(f'{pic_results_path}/sub-{subj}/sub-{subj}_task-pictures_betas.pickle')
    vid_betas = pd.read_pickle(f'{vid_results_path}/sub-{subj}/sub-{subj}_task-video_remap-1_betas.pickle')
   
    for roi in rois:
        #returns indices for left and right hemisphere (mmp[0] and mmp[1] respectively)
        mmp = get_rois(roi, glasserpth='./glasser_et_al_2016')
        # Stack voxels across hemispheres before MVPA
        nvert = sum(mmp[0]) + sum(mmp[1])
        #subselect voxels for each
        pic_betas_roi = np.vstack([pic_betas[h][mmp[hi], :] for hi, h in enumerate(['L','R']) ])
        vid_betas_roi = np.vstack([vid_betas[h][mmp[hi], :] for hi, h in enumerate(['L','R']) ])

        mean_pic_B = np.mean(pic_betas_roi.reshape((-1,n_pics,3)), axis=2)
        mean_vid_B = np.mean(vid_betas_roi.reshape((-1,n_vids,2)), axis=2)

        mean_pic_B = pd.DataFrame(mean_pic_B, columns=pic_conditions)
        mean_vid_B = pd.DataFrame(mean_vid_B,columns=vid_conditions)

        if roi == 'earlyvisual':
            evc_pic_Bs[idx] = mean_pic_B
            evc_vid_Bs[idx] = mean_vid_B
        elif roi == 'ventralvisual':
            vvc_pic_Bs[idx] = mean_pic_B
            vvc_vid_Bs[idx] = mean_vid_B
        elif roi == 'scene-occipital':
            scene_pic_Bs[idx] = mean_pic_B
            scene_vid_Bs[idx] = mean_vid_B

plot_distributions = False

for roi in rois:
    if roi == 'earlyvisual':
        pic_arr = evc_pic_Bs
        vid_arr = evc_vid_Bs
    elif roi == 'ventralvisual':
        pic_arr = vvc_pic_Bs
        vid_arr = vvc_vid_Bs
    elif roi == 'scene-occipital':
        pic_arr = scene_pic_Bs
        vid_arr = scene_vid_Bs

    #try averaging response patterns before doing the correlation
    avg_vid_response = pd.DataFrame(np.mean(vid_arr, axis=0), columns=vid_conditions)
    avg_pic_response = pd.DataFrame(np.mean(pic_arr, axis=0), columns=pic_conditions)
    
    for vid in ['moana.mp4']:
        vid_response = avg_vid_response[vid]
        pearson_df = pd.DataFrame(index=[0],columns=pic_vid_mapping[vid])
        for pic in pic_vid_mapping[vid]:
            pic_response = avg_pic_response[pic]
            pearson_df.loc[0,pic]=np.corrcoef(vid_response,pic_response)[0,1]
        
        pearson_df.to_csv(f'pic-corrs/across-subj_roi-{roi}_vid-{vid}_pearson.csv')
    
results = {vid:pd.DataFrame(index=subjlist, columns=pic_vid_mapping[vid]) for vid in vid_conditions}
for idx, subj in enumerate(subjlist):
    vid_df = pd.DataFrame(vid_arr[idx,:,:],columns=vid_conditions)
    pic_df = pd.DataFrame(pic_arr[idx,:,:],columns=pic_conditions)
    for vid in vid_conditions:
        vid_response = vid_df[vid]
        pearson_df = pd.DataFrame(index=[0],columns=pic_vid_mapping[vid])
        for pic in pic_vid_mapping[vid]:
            pic_response = pic_df[pic]
            pearson_df.loc[0,pic]=np.corrcoef(vid_response,pic_response)[0,1]
        results[vid].loc[subj] = pearson_df.loc[0]

for vid, df in results.items():
    # save the results
    df.to_csv(f'roi-{roi}_{vid}-pic_pearson-correlation.csv')

    mean_df = df.mean(axis=0)
    top_two = mean_df.nlargest(2)
    print(
        f"""
        In ROI {roi} movie {vid} correlations:
        {top_two.index[0]}={top_two.iloc[0]} ; {top_two.index[1]}={top_two.iloc[1]}
        """
    )

if plot_distributions:
    for vid, df in results.items():
        dfm = df.melt(var_name='columns')
        g = sns.FacetGrid(dfm, col='columns')
        g = (g.map(sns.distplot, 'value'))

        plt.savefig(f'./img_comp_figs/roi-{roi}_video-{vid}_pearson-corr.png')

   


