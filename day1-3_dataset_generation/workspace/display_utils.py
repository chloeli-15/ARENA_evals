# Display markdown in IPython
import IPython.display


def display_markdown(text):
    """
    Display markdown text in IPython.

    Args:
        text (str): Markdown formatted text to display
    """

    IPython.display.display(IPython.display.Markdown(text))


def display_markdown_and_code(text):
    """
    Display markdown text in IPython and as literal (easy for copy paste)

    Args:
        text (str): Markdown formatted text to display
    """
    # escape xml tags, which are commonly used in prompts
    text_with_escaped_xml_tags = text
    for original_string, replacement_string in [("<", "\<"), (">", "\>")]:
        text_with_escaped_xml_tags = text_with_escaped_xml_tags.replace(
            original_string, replacement_string
        )
    display_markdown(text_with_escaped_xml_tags)

    # now show the literal question, how the model sees it
    display_markdown("```\n" + text + "\n```")
