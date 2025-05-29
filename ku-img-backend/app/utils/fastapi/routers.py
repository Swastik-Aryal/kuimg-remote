from typing import Any, Callable

from fastapi import APIRouter as FastAPIRouter
from fastapi.types import DecoratedCallable

# CustomerAPIRouter1
# Why? 
# A problem we will face in fastapi router is that "/endpoint/" redirects to "/endpoint" (and vice versa)
# Normally this wouldn't be an issue, however during https deployments this will cause problems.
# Previously we could sort of "hack" it by using "/endpoint/?", but that no longer works with the update.
# Hence we define a custom API Router so that redirects don't occur.

# thread : https://github.com/tiangolo/fastapi/issues/2060
# src : https://github.com/tiangolo/fastapi/issues/2060#issuecomment-834868906
# the code has been slightly modified.
# 
# Basically, the code creates both the routes (however only 1 is documented in docs)

class CustomAPIRouter1(FastAPIRouter):
    def api_route(
        self, 
        path: str, 
        *, 
        include_in_schema: bool = True, 
        **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:

        path = path.rstrip("/")
        alternate_path = path + "/"

        add_path = super().api_route(path, include_in_schema=include_in_schema, **kwargs)
        add_alternate_path = super().api_route(alternate_path, include_in_schema=False, **kwargs)

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            add_alternate_path(func)
            return add_path(func)

        return decorator
