push!(LOAD_PATH,"../")
using Documenter
using DocStringExtensions

@template DEFAULT =
    """
    $(DOCSTRING)
    """

@template TYPES =
    """
    $(TYPEDEF)
    $(DOCSTRING)
    """

@template (METHODS, MACROS) =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    $(METHODLIST)
    """


makedocs()
