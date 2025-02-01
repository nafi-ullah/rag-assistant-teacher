from langchain_community.tools.tavily_search import TavilySearchResults


def get_profile_url_tavily(name: str):
    """Searches for Linkedin or Twitter Profile Page."""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res



# And now we want to write a function that its entire purpose is to get a name and find its LinkedIn URL.

# And for that, we're going to be using a third party, which is called Tivoli.

# And Tivoli has a nice integration with link Chain.

# So Tivoli Is an API, a search API which is highly optimized for generative AI workloads.

# So if we're using LLM agents like we're doing right now, or building Rag application retrieval augmentation

# generation, which we do in the second part of the course.

# So this search engine is highly optimized for those kind of applications.

# So it not only uses those search engines like Google and Bing, but it also has pre-built implemented

# logic of taking our questions and figuring out what is the best answer that we're looking for.

# That would suit us for our generative AI application.

# So I think the best way is to show you by an example, and you can see it has a very generous free tier