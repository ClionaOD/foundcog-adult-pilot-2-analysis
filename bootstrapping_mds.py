"""
Implementation of Bootstrapping MDS Solutions by Jacoby and Armstrong 2013

@author: Cliona O'Doherty
"""

import numpy as np
from numpy.linalg import cond
import pandas as pd
import pickle
import warnings
import random
import os

import scipy.spatial.distance as ssd
from scipy.spatial import procrustes
from sklearn.manifold import MDS

#from model_design_matrix import get_design_matrix

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import pathlib

#from contrast_list import get_con_list

warnings.filterwarnings("ignore")


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', edgecolor='blue', label='', **kwargs):
    """
    Copied from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    ----------
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, label=label, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def construct_rdm(observation_df, metric='correlation'):
    """
    Takes a n*k df with n observations and k conditions,
    returns the data as an n*n rdm dataframe
    """
    rdm = ssd.pdist(observation_df.values.T, metric=metric)
    rdm = ssd.squareform(rdm)
    rdm = pd.DataFrame(rdm, columns=observation_df.columns,
                       index=observation_df.columns)

    return rdm


def get_mds_embedding(rdm, ref=None):
    """
    returns a k*m dataframe with k observations and m dimensions

    ref: a reference mds embedding to which the returned embedding will be aligned (if provided)
    """
    mds = MDS(n_components=2, dissimilarity='precomputed')
    df_embedding = pd.DataFrame(mds.fit_transform(rdm.values), index=rdm.index)

    if ref is not None:
        # Some random combinations of movies may drop a condition (this is rare).
        # Check that both are the same shape, fill in NaN where not.
        if not df_embedding.shape == ref.shape:
            df_embedding = df_embedding.reindex(index=ref.index)
            df_embedding = df_embedding.fillna(0)
        mtx1, mtx2, disparity = procrustes(ref, df_embedding.values)
        df_embedding = pd.DataFrame(mtx2, index=rdm.index)

    return df_embedding


def bootstrapped_mds(rdms, q=50, con_names=None, reference=None, across_subject_average_type='median'):
    """
    takes events, constructs resampled design matrix, performs bootstrapped mds

    events: dict with keys=movie_names and values=events_files
    q: number of iterations for bootstrapping
    con_list: optional list of contrasts, in form of list of dict/str like this [{'animate':1, 'inanimate':-1}, {'open':1, 'outside':1}, 'social'} where 'social' is equvialent to {'social':1}....]
    reference:    dataframe for Procrustes transform

    """

    bootstrap_embeddings = []

    for i in range(q):
        # 1. sample n rows with replacement from V (observation matrix)
        # Sample w replacement from RDM subjlist
        rdms_q = random.choices(rdms, k=len(rdms))

        # 2. get average across subjects
        rdms_across_subj=np.stack([rdm for rdm in rdms_q], axis=2).transpose([2,0,1]) # subjects in axis 0                 

        # Average across subjects
        # How should we average RDMs across subjects for summary measure and during bootstrapping            
        if across_subject_average_type=='mean':
            avfunc=np.mean
        elif across_subject_average_type=='median':
            avfunc=np.median
        rdms_across_subj_average=avfunc(rdms_across_subj, axis=0)

        rdms_across_subj_average = pd.DataFrame(rdms_across_subj_average, index=con_names, columns=con_names)

        # 3. perform MDS on the rdm for this iteration
        if i == 0:
            if reference is None:
                reference = get_mds_embedding(rdms_across_subj_average)

            mds_q = get_mds_embedding(rdms_across_subj_average, ref=reference)
        else:
            mds_q = get_mds_embedding(rdms_across_subj_average, ref=reference)

        bootstrap_embeddings.append(mds_q)

    # 4. restructure the data
    bootstrapped_coords = {}
    for k in con_names:
        # create X, a q*m matrix of bootstrapped coordinates for the condition/object
        X = []
        for mds_q in bootstrap_embeddings:
            x_i = mds_q.loc[k].values
            X.append(x_i)
        bootstrapped_coords[k] = np.array(X)

    return bootstrapped_coords, con_names, reference


def bootstrap_mds_across_subjects(con_list=None,
                  rdms=None,
                  ax=None,
                  use_precomputed_bootstrap=False,
                  use_precomputed_reference=False,
                  nreferences=1,
                  nbootstraps_per_reference=1,
                  ninitial=20,
                  initial_q=50,
                  q=1000,
                  n_std=1.0, 
                  save_results=False,
                  save_figures=False,
                  test_type=None,
                  outpath='.',
                  figpath='.'):
    """
    Function copied from FOUNDCOG modelling. OG function uses events files rather than RDMs
    """
    results = []

    for reference_ind in range(nreferences):
        if use_precomputed_reference:
            with open(f'./{test_type}_bootstrap_mds_reference_coords_q_{q}_nstd_{n_std}.pickle', 'rb') as f:
                reference = pickle.load(f)
        else:
            # Do a set of iterations and find one with lowest variance (so a good reference)
            print('Searching for good reference for MDS')
            reference_min = None
            bootstrap_var_min = np.inf
            for perm in range(ninitial):
                # CHANGE #
                bootstrap_coords, con_names, reference = bootstrapped_mds(
                    rdms, q=initial_q, con_names=con_list)
                bootstrap_var = 0
                for condition, coords_arr in bootstrap_coords.items():
                    bootstrap_var += coords_arr.var(axis=0).sum()
                if bootstrap_var < bootstrap_var_min:
                    reference_min = reference
                    bootstrap_var_min = bootstrap_var

            reference = reference_min
            print('Found MDS reference')
            with open(f'./{test_type}_bootstrap_mds_reference_coords_q_{q}_nstd_{n_std}.pickle', 'wb') as f:
                pickle.dump(reference, f)

        for bootstrap_ind in range(nbootstraps_per_reference):
            if use_precomputed_bootstrap:
                with open(f'./{test_type}_bootstrap_mds_coords_q_{q}_nstd_{n_std}.pickle', 'rb') as f:
                    bootstrap_coords = pickle.load(f)
            else:
                # Use this reference for the main bootstrap
                # CHANGE #
                bootstrap_coords, con_names, reference = bootstrapped_mds(
                    rdms, q=q, con_names=con_list, reference=reference)
                with open(f'./{test_type}_bootstrap_mds_coords_q_{q}_nstd_{n_std}.pickle', 'wb') as f:
                    pickle.dump(bootstrap_coords, f)

            cmap = plt.get_cmap('hsv')
            colors = cmap(np.linspace(0, 1.0, len(bootstrap_coords)+1))

            _, ax = plt.subplots(ncols=1, figsize=(11.69, 8.27))

            bootstrap_var = 0
            for idx, (condition, coords_arr) in enumerate(bootstrap_coords.items()):
                x = coords_arr[:, 0]
                y = coords_arr[:, 1]
                # ax.scatter(x,y,s=0.01)
                confidence_ellipse(x, y, ax=ax, n_std=n_std,
                                   edgecolor=colors[idx], label=condition)

                np.mean(coords_arr)

                ax.text(np.mean(x), np.mean(y), condition,
                         ha='center', va='center', fontsize='x-small')
                bootstrap_var += coords_arr.var(axis=0).sum()

            results.append({'reference_ind': reference_ind,
                            'bootstrap_ind': bootstrap_ind,
                            'ninitial': ninitial,
                            'initial_q': initial_q,
                            'q': q,
                            'con_list': con_list,
                            'bootstrap_var': bootstrap_var})
            print(
                f'Reference {reference_ind} iteration {bootstrap_ind} final bootstrap variance {bootstrap_var}')

            ax.set_xlim((-0.35, 0.35))
            ax.set_ylim((-0.35, 0.35))
            ax.set_aspect('equal')
            ax.axis('off')

            h, l = ax.get_legend_handles_labels()
            ax.axis('off')
            legend = ax.legend(h, l,  prop={'size': 5}, loc='upper right')
            plt.tight_layout()

            if save_figures:
                plt.savefig(os.path.join(figpath,f'bootstrap/{test_type}_bootstrap_results_q_{q}_std_{n_std}_ninitial_{ninitial}_initialq_{initial_q}_ref_{reference_ind}_iteration_{bootstrap_ind}.pdf'))

    if save_results:
        with open(os.path.join(outpath,f'bootstrap/{test_type}_bootstrap_results_q{q}_std_{n_std}_ninitial_{ninitial}_initialq_{initial_q}.pickle'), 'wb') as f:
            pickle.dump(results, f)

    return results

if __name__ == "__main__":
    # Data path, not repo path
    analysis_version='v1-noconfounds'
    fap='/home/cusackrh/foundcog-adult-pilot/{analysis_version}'
    figpath = f'{pathlib.Path(__file__).parent.resolve()}/{analysis_version}/figs/bootstrap'
    outpath = os.path.join(fap, analysis_version, 'bootstrap')

    tosubtract = 'voxelmean'
    tasklist = ['pictures','video','picvid']
    subjlist = [f'{subjind:02}' for subjind in range(2,18)]
    hemilist = ['both']
    rois = ['earlyvisual','ventralvisual']
    #conditions = ['pingu.mp4', 'moana.mp4', 'brother_bear.mp4', 'bathsong.mp4', 'playground.mp4', 'rio_jungle_jig.mp4', 'bedtime.mp4', 'kids_kitchen.mp4', 'ratatouille.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'real_cars.mp4', 'new_orleans.mp4', 'dog.mp4', 'piper.mp4', 'our_planet.mp4']
    conditions = ['pingu', 'moana', 'brother_bear', 'bathsong', 'playground', 'rio_jungle', 'bedtime', 'kids_kitchen', 'ratatouille', 'minions', 'forest', 'real_cars', 'new_orleans', 'dog', 'piper', 'our_planet']
    
    matrix_cols = [
        [f'{mov}_vid-1' for mov in conditions],
        [f'{mov}_vid-2' for mov in conditions],
        [f'{mov}_pic-1' for mov in conditions],
        [f'{mov}_pic-2' for mov in conditions],
        [f'{mov}_pic-3' for mov in conditions]
    ]
    matrix_cols = [item for sublst in matrix_cols for item in sublst]

    for roi in rois:
        all_subj_rdms = [[],[],[]]

        # Load the compiled 80*80 RDMs (5 runs * 16 conditions) for each subject, average across
        # hemispheres, calculate the average RDMs for each task (pictures, video, picvid), and add them
        # to the full list of all RDMs.
        for subj in subjlist:
            fullrdms_byhemi = []
            for hemiind,hemi in enumerate(hemilist):
                rdm_onehemi = pd.read_pickle(f'{fap}/results/sub-{subj}/sub-{subj}_task-combined_hemi-{hemi}_roi-{roi}_remap-1_rdms_subtract-{tosubtract}.pickle')
                fullrdms_byhemi.append(rdm_onehemi)
            fullrdm = np.mean(np.array(fullrdms_byhemi),axis = 0)
            Q = [M for SubA in np.split(fullrdm,5, axis = 0) for M in np.split(SubA,5, axis = 1)] 
            for taskind,task in enumerate(tasklist):
                if task == 'pictures':
                    rdm = np.mean(np.stack((Q[1],Q[2],Q[5],Q[7],Q[10],Q[11]),axis = 2), axis = 2)
                elif task == 'video':
                    rdm = np.mean(np.stack((Q[19],Q[23]),axis = 2), axis = 2)
                elif task == 'picvid':
                    rdm = np.mean(np.stack((Q[3],Q[4],Q[8],Q[9],Q[13],Q[14],Q[15],Q[16],Q[17],Q[20],Q[21],Q[22]),axis = 2), axis = 2)

                all_subj_rdms[taskind].append(rdm)
        

        # Give this function a list of RDMs (rdms, one per subject in this case) with whatever the dataframe columns should be (con_list)
        # Provide test_type for saving bootstrap results so as not to overwrite
        os.makedirs(figpath, exist_ok=True)
        os.makedirs(outpath, exist_ok=True)
        results_pic = bootstrap_mds_across_subjects(con_list=conditions, rdms = all_subj_rdms[0], test_type=f'{roi}_pictures_',save_figures=True, figpath = figpath, outpath=outpath)
        results_vid = bootstrap_mds_across_subjects(con_list=conditions, rdms = all_subj_rdms[1], test_type=f'{roi}_video_',save_figures=True, figpath = figpath, outpath=outpath)
        results_picvid = bootstrap_mds_across_subjects(con_list=conditions, rdms = all_subj_rdms[2], test_type=f'{roi}_picvid_',save_figures=True, figpath = figpath, outpath=outpath)

