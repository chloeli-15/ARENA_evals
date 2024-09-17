import pathlib

import seaborn as sns
import matplotlib.pyplot as plt

# output directory to save plots to
OUTPUT_IMAGES_DIR = pathlib.Path("images")


def _get_image_filepath(filename: str) -> pathlib.Path:
    """
    Get the filepath to save a plot to
    """

    current_dir = pathlib.Path(__file__).parent.resolve()

    return current_dir / OUTPUT_IMAGES_DIR / filename


def save_current_plot_to_file(filename: str):
    """
    Save the current plot to a file in images directory
    """

    filepath = _get_image_filepath(filename)

    print(f"Saving to {filepath}...")

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {filepath}")


def save_seaborn_figure_to_file(fig: sns.FacetGrid, plot_filename: str) -> None:
    """
    Save a seaborn figure to a file in images directory
    """

    filepath = _get_image_filepath(plot_filename)

    print(f"Saving to {filepath}...")

    fig.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {filepath}")
