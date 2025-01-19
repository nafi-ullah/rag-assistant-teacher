from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser
# So we want to import list and any. So we want to use for typing. And we want to use the pedantic output parser. So what's pedantic in a nutshell?
# It's very similar to the built in data class in Python, but it's actually an external library that is helping us with data validation and setting management. So it will allow us to define schemas and to validate inputs against those schemas.

from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)