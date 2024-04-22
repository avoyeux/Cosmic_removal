import re 
import os

import numpy as np

from typeguard import typechecked


class RE:
    """
    Contains multipurpose functions related to the re library.
    """

    @staticmethod
    @typechecked
    def replace_group(pattern_match: re.Match[str], groupname: str, new_value: str) -> str:
        """
        To replace only one group value of a pattern match object.
        """

        start, end = pattern_match.span(groupname)
        return pattern_match.string[:start] + new_value + pattern_match.string[end:]