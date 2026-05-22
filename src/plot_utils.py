import os
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

sns.set(font_scale=0.8)


def _save_or_show(fig, output_dir, filename):
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _plot_attn_on_axis(axis, head_weights, tokens, title, cmap='viridis',
                       xlabel='Key Tokens', ylabel='Query Tokens', vmin=None, vmax=None):
    display_tokens = [t.replace(config.TOK_DELIM, '') for t in tokens]
    sns.heatmap(
        head_weights, cmap=cmap,
        xticklabels=display_tokens, yticklabels=display_tokens,
        ax=axis, square=True, cbar_kws={'shrink': 0.8},
        vmin=vmin, vmax=vmax,
    )
    axis.set_title(title, fontsize=10)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=10)


def plot_attn_head(head_weights, tokens, title, cmap='viridis', output_dir=None, filename=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    _plot_attn_on_axis(ax, head_weights, tokens, title, cmap)
    plt.tight_layout()
    _save_or_show(fig, output_dir, filename)
