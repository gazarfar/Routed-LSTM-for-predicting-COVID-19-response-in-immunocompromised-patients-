""" Summary plots of SHAP values across a whole dataset.
"""

import warnings

import matplotlib.pyplot as pl
import numpy as np



from shap.plots import colors
from shap.plots._labels import labels


myorder = None
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def My_beeswarm(shap_values,shap_values_plot, max_display=10,
             features=None, feature_names = None, color=colors.red_blue,
             axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, color_bar_label=labels["FEATURE_VALUE"]):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    color_bar : bool
        Whether to draw the color bar (legend).

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged.

    Examples
    --------

    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """

    
   
    
    # if out_names is None: # TODO: waiting for slicer support
    #     out_names = shap_exp.output_names
    mean_Shap_values = np.mean(np.abs(shap_values), axis = 0)
    order = np.flip(np.argsort(mean_Shap_values))
    
    if log_scale:
        pl.xscale('symlog')

   


    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]

     
    feature_order = order[0:max_display]
        
    yticklabels = [feature_names[i] for i in feature_order]
    
    row_height = 0.4

    pl.gcf().set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
   
    pl.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_order)):
        pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values_plot[:, i]
        values = features[:,i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
       
        shaps = shaps[inds]
        values = values[inds]
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        # print(layer)
        ys *= 0.9 * (row_height / np.max(ys + 1))
        # print(ys.shape)
        
        # trim the color range, but prevent the color range from collapsing
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)
        if vmin > vmax: # fixes rare numerical precision issues
          vmin = vmax
     

        # plot the non-nan fvalues colored by the trimmed feature value
        cvals = values.astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
        cvals[cvals_imp > vmax] = vmax
        cvals[cvals_imp < vmin] = vmin
        pl.scatter(shaps, pos + ys,
                   cmap=color, vmin=vmin, vmax=vmax, s=16,
                   c=cvals, alpha=alpha, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)
       

    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=color)
    m.set_array([0, 1])
    cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    cb.set_label(color_bar_label, size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), reversed(yticklabels), fontsize=13)
    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    pl.xlabel(labels['VALUE'], fontsize=13)
    pl.show()
    
    return mean_Shap_values, order





