# Documentation Standards

High-quality documentation is a signature of the **MolSysSuite**. ElastNetMT follows the **NumPy Style** for all Python docstrings.

## Docstring Structure

All public methods must follow this template:

```python
def my_method(arg1, arg2=None):
    """
    Brief one-sentence description.

    Detailed explanation of the algorithm, assumptions, or physical 
    context. Link to theoretical foundations if needed.

    Parameters
    ----------
    arg1 : type
        Description of arg1.
    arg2 : type, optional
        Description of arg2. (Default: None).

    Returns
    -------
    type
        Description of the return value.

    Examples
    --------
    >>> model = GNM('protein.pdb')
    >>> model.my_method(val)
    
    See Also
    --------
    other_method : Related functionality.
    """
    pass
```

## Diagrams and Math

- **Mermaid:** Use Mermaid blocks (```mermaid) for architecture and workflows.
- **LaTeX:** Use double dollar signs ($$) for mathematical equations.
- **MyST-NB:** Notebooks for user tutorials should be executable and use MyST markdown for a clean build.

## Internal Docstrings
Private methods (prefixed with `_`) should also be documented, but in a simpler format, focusing on developer intent rather than user guidance.
