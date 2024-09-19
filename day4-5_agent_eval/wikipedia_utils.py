import wikipedia

import re
import dataclasses

WikiPageTitleStr = str
WikiPageLinkStr = str
WikiPageSummaryStr = str
WikiPageContentStr = str


def show_basic_page_info(page: wikipedia.WikipediaPage):
    print("Title:", page.title)
    print("\nURL", page.url)
    print(f"\nSummary (word count {len( page.summary.split())}):", page.summary)
    print(
        f"\nContent (word count {len( page.content.split())}):",
        page.content[:1000],
        "......",
    )
    print(f"""\nLinks (link count {len(page.links)}): [{", ".join(page.links[:7])}, ......]""")


# Retrieve a Wikipedia page from its title
# page = wikipedia.page("Large language model")
# show_basic_page_info(page)


def get_page(title: WikiPageTitleStr) -> wikipedia.WikipediaPage:
    """
    Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option.
    If the title is not found, try to find a similar title.

    Args:
        title (str): The title of the Wikipedia page

    Returns:
        WikipediaPage: The Wikipedia page
    """
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    except wikipedia.DisambiguationError as e:
        # ex: page = wikipedia.page("Python")
        #
        # goes to the first disambiguation, so some are likely unreachable?
        #
        return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
    except wikipedia.PageError as e:
        # ex: page = wikipedia.page("Animalss", auto_suggest=False)
        return wikipedia.page(title, auto_suggest=True, redirect=True)


def get_permitted_links(current_page: wikipedia.WikipediaPage) -> list[WikiPageLinkStr]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    all_links = current_page.links
    content = current_page.content
    permitted_links = [link for link in all_links if link in content]

    # note: why do we have a self reference?
    if current_page.title in permitted_links:
        permitted_links.remove(current_page.title)

    return permitted_links


def is_permitted_link(current_page: wikipedia.WikipediaPage, link: WikiPageLinkStr) -> bool:
    return link.lower() in (x.lower() for x in get_permitted_links(current_page))


def get_page_summary(page: wikipedia.WikipediaPage) -> WikiPageSummaryStr:
    """
    Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

    Args:
        page (WikipediaPage): The Wikipedia page object.

    Returns:
        str: The summary of the Wikipedia page.
    """
    page = page if page else self.goal_page
    summary = page.content[:500]
    last_period_index = summary.rfind(".")
    return summary[: last_period_index + 1] if last_period_index != -1 else summary


def get_page_content(page: wikipedia.WikipediaPage) -> WikiPageContentStr:
    """
    Get all the content for the wikipedia page you are currently on.

    Anything which corresponds to a link is wrapped in <link></link> tags.

    Example:

        Barack Hussein <link>Obama</link> II (born August 4, 1961) is an American politician who
        served as the 44th <link>president of the United States</link> from 2009 to 2017. As a member of the Democratic
        Party, he was the first African-American  president in U.S. history. Obama previously served as a <link>U.S.
        senator</link> representing <link>Illinois</link> from 2005 to 2008 and as an <link>Illinois state senator</link>
        from 1997 to 2004.
        Obama was born in <link>Honolulu, Hawaii</link>. He graduated from <link>Columbia University</link> in 1983 with a
        <link>Bachelor of Arts</link> degree in political science and later worked as a <link>community organizer</link> in
        <link>Chicago</link>. In 1988, Obama would enroll in <link>Harvard Law School</link>, where he became the first
        black president of the <link>Harvard Law Review</link>. He became a civil rights attorney and an academic, teaching
        constitutional law at the <link>University of Chicago Law School</link> from 1992 to 2004. He also went into
        elective politics; Obama represented the 13th district in the <link>Illinois Senate</link> from 1997 until 2004,
        when he successfully ran for the U.S. Senate. In the 2008 presidential <link>election</link>, after a close primary
        campaign against <link>Hillary Clinton</link>, he was nominated by the Democratic Party for president. Obama
        selected <link>Joe Biden</link> as his running mate and they defeated Republican nominees <link>John McCain</link>
        and <link>Sarah Palin</link>.

        ...

        == See also ==


        === Politics ===
        <link>DREAM Act</link>
        <link>Fraud Enforcement and Recovery Act of 2009</link>
        <link>Immigration Reform and Control Act of 1986</link>
        <link>IRS targeting controversy</link>
        <link>Middle Class Tax Relief and Job Creation Act of 2012</link>
        <link>National Broadband Plan (United States)</link>
        <link>Office of Energy Efficiency and Renewable Energy</link>
        <link>Social policy of the Barack Obama administration</link>
        <link>SPEECH Act</link>
        <link>Stay with It</link>
        <link>White House Office of Energy and Climate Change Policy</link>


        === Other ===
        <link>Roberts Court</link>
        <link>Speeches of Barack Obama</link>

    """

    content = page.content

    permitted_links = get_permitted_links(page)

    for word in sorted(permitted_links, key=len, reverse=True):

        content = re.sub(
            r"""(\s|[,.)!?;:'"])(""" + re.escape(word) + r""")(\s|[,.)!?;:'"s])""",
            r"\1<link>\2</link>\3",
            content,
            count=1,
            flags=re.IGNORECASE,
        )

    return content
