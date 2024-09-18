import pathlib

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

# Set the visual style of the plots
sns.set(style="whitegrid")

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


def plot_barplot_of_model_responses(
    df: pd.DataFrame,
    model_column: str,
    score_column: str,
    groupby_column: str,
) -> None:

    # Group the dataframe by the specified column
    grouped = df.groupby(groupby_column)

    # Calculate number of subplots needed (one row per group)
    n_groups = len(grouped)

    # Create a figure with subplots (one row per group)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 6 * n_groups), squeeze=False)
    axes = axes.flatten()

    for (group_name, group_data), ax in zip(grouped, axes):
        # Sort values by model, then parsed_response for this group
        group_data = group_data.sort_values(by=[model_column, score_column])

        # Create a countplot for this group
        sns.countplot(data=group_data, x=score_column, hue=model_column, palette="Set2", ax=ax)

        # Add titles and labels for each subplot
        ax.set_title(f"{groupby_column}: {group_name}", fontsize=16)
        ax.set_xlabel("Parsed Response", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)

        # Set x-axis limits to 0 to 4
        ax.set_xlim(0, 4)

        # Adjust legend title
        ax.legend(title="Model")

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_violin_of_model_responses(
    df: pd.DataFrame,
    model_column: str,
    score_column: str,
    groupby_column: str,  # Argument for grouping
) -> None:
    # Group the dataframe by the specified column
    grouped = df.groupby(groupby_column)

    # Calculate number of subplots needed (one row per group)
    n_groups = len(grouped)

    # Create a figure with subplots (one row per group)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 6 * n_groups), squeeze=False)
    axes = axes.flatten()

    for (group_name, group_data), ax in zip(grouped, axes):
        # Sort the data for this group
        group_data = group_data.sort_values(by=[model_column, score_column])

        # Create violin plot for this group
        sns.violinplot(
            y=model_column, x=score_column, data=group_data, hue=model_column, palette="Set2", ax=ax
        )

        # Set labels and title for each subplot
        ax.set_ylabel("Model")
        ax.set_xlabel("Parsed Response")
        ax.set_title(f"{groupby_column}: {group_name}")

        # Set x-axis limits to 0 to 4
        ax.set_xlim(0, 4)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
