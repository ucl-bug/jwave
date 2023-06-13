import inspect
import re

from griffe.dataclasses import Docstring
from griffe.docstrings.dataclasses import (
    DocstringSectionParameters,
    DocstringSectionReturns,
    DocstringSectionText,
)
from griffe.docstrings.parsers import Parser, parse
from markdown import markdown
from plum.function import Function
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


class Implementation(object):
    def __init__(self, name, params, docs):
        self.name = name
        self.params = params
        self.docs = self.parse_docs(docs)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = self._signature
        string += "\n\n"
        string += self.docs if self.docs else ""
        string += "\n"
        return string

    def parse_docs(self, docs):
        # Extracting text and parameters
        docstring = Docstring(docs)
        parsed = parse(docstring, Parser.google)

        text = [x for x in parsed if isinstance(x, DocstringSectionText)][0].value
        try:
            params = [x for x in parsed if isinstance(x, DocstringSectionParameters)][0]
        except:
            params = None

        try:
            returns = [x for x in parsed if isinstance(x, DocstringSectionReturns)][0]
        except:
            returns = None

        # Transform parameters into table
        if params:
            text += "<p><strong>Parameters:</strong></p>"
            table = '<div class="md-typeset__scrollwrap"><div class="md-typeset__table"><table>'
            table += "<thead><tr><th>Name</th><th>Type</th><th>Description</th><th>Default</th></tr></thead>"
            table += "<tbody>"

            for p in params.value:
                table += "<tr>"
                table += f"<td><code>{p.name}</code></td>"

                if p.annotation:
                    table += f"<td><code>{p.annotation}</code></td>"
                else:
                    table += "<td></td>"
                if p.description:
                    table += f"<td>{markdown(p.description)}</td>"
                else:
                    table += "<td></td>"

                # Get default value
                default = self.params[p.name]._default
                this_param_default = (
                    "<em>required</em>"
                    if default == inspect._empty
                    else f"<code>{default}</code>"
                )
                table += f"<td>{this_param_default}</td>"
                table += "</tr>"

            table += "</tbody></table></div></div>"
            text += table

        # Transform returns into table
        if returns:
            text += "<p><strong>Returns:</strong></p>"
            table = '<div class="md-typeset__scrollwrap"><div class="md-typeset__table"><table>'
            table += "<thead><tr><th>Type</th><th>Description</th></tr></thead>"
            table += "<tbody>"

            for r in returns.value:
                table += "<tr>"
                table += f"<td><code>{r.name}</code></td>"
                if r.description:
                    table += f"<td>{markdown(r.description)}</td>"
                else:
                    table += "<td></td>"
                table += "</tr>"
            table += "</tbody></table></div></div>"
            text += table

        return text

    @staticmethod
    def param_to_string(names, types, defaults):
        namestring = f"{names}"
        typestring = ": " + strip_modules(str(types)) if types != inspect._empty else ""
        defaultstring = f" = {defaults}" if defaults != inspect._empty else ""
        return namestring + typestring + defaultstring

    @property
    def _signature(self):
        string = f"{self.name}("
        counter = 0
        for n in self.params.keys():
            thistype = self.params[n]._annotation
            default = self.params[n]._default
            string += ", " if counter > 0 else ""
            string += self.param_to_string(n, thistype, default)
            counter += 1
        string += ")"
        string = highlight(string, PythonLexer(), HtmlFormatter())

        string = var_to_bold(string)

        # Change outer <div> form class "highlight" to class "highlight language-python
        string = string.replace(
            '<div class="highlight">', '<code class="highlight language-python">'
        )
        # Remove <pre> and </pre>
        string = string.replace("<pre>", "").replace("</pre>", "")
        # Update <div> closing tag
        string = string.replace("</div>", "</code>")

        # Wrap around various containers
        string = '<h3 class="doc doc-heading">' + string + "</h3>"

        return string


def var_to_bold(string):
    # Make variable names in the string bold
    # example "... <span class="n">text</span><span class="p">:</span> ..."
    # becomes
    # "... <span class="n"><strong>text_value</strong></span><span class="p">:</span> ..."
    string = re.sub(
        r'<span class="n">([^<]+)</span><span class="p">:</span>',
        r'<span class="n"><strong>\1</strong></span><span class="p">:</span>',
        string,
    )
    string = re.sub(
        r'<span class="n">([^<]+)</span> <span class="o">=</span>',
        r'<span class="n"><strong>\1</strong></span> <span class="o">=</span>',
        string,
    )
    string = re.sub(
        r'<span class="n">([^<]+)</span><span class="p">,</span>',
        r'<span class="n"><strong>\1</strong></span><span class="p">,</span>',
        string,
    )

    return string


def strip_modules(string):
    # Removes the module roots and the <class > tags from the string.
    # Example:
    # typing.Union[jwave.geometry.MediumObject[<class 'object'>, <class 'object'>, <class 'jaxdf.discretization.OnGrid'>], jwave.geometry.MediumObject[<class 'object'>, <class 'jaxdf.discretization.OnGrid'>, <class 'object'>]]
    # becomes
    # Union[MediumObject[object, object, OnGrid], MediumObject[object, OnGrid, object]]
    to_change = {
        "typing.Union": "Union",
        "jwave.geometry.MediumObject": "Medium",
        "<class ": "",
        ">": "",
        "jaxdf.discretization.": "",
        "jwave.geometry.": "",
        "jaxdf.core.": "",
    }
    for key, value in to_change.items():
        string = string.replace(key, value)
    return string


def _extract_implementations(plum_func):
    name, function = plum_func
    implementations = []

    for signature, method in zip(function.methods.keys(), function.methods.values()):
        method = method[0].__wrapped__
        params = inspect.signature(method).parameters
        docs = inspect.getdoc(method)

        implementations.append(Implementation(name, params, docs))

    # Remove implementations with the same _signature
    # TODO: Understand why those duplicates exist at all..
    impl = {}
    for i in implementations:
        signature = i._signature
        if signature not in impl.keys():
            impl[signature] = i

    # Sort dictionary by key
    impl = {k: v for k, v in sorted(impl.items(), key=lambda item: item[0])}

    implementations = list(impl.values())
    return implementations


def mod_to_string(module_name, function=None):
    module_chain = module_name.split(".")
    mod, submodules = module_chain[0], module_chain[1:]
    # Get functions variables generated by the module
    mod = __import__(mod)
    for att in submodules:
        mod = getattr(mod, att)
    operators = set(inspect.getmembers(mod, lambda x: isinstance(x, Function)))

    if function is not None:
        # Keep only the operators whose first element is the same as the function
        operators = [op for op in operators if op[0] == function]

    # Sort operators by name
    def keyfun(op):
        func = op[1].methods
        func = func[func.keys()[0]][0].__wrapped__.__name__
        return func

    operators = [v for v in sorted(operators, key=lambda item: item[0])]

    implementations = list(map(_extract_implementations, operators))

    # Concatenate all implementations into a string
    text = ""
    for fun in implementations:
        # text += f'## `{fun[0].name}`\n\n'
        for i in fun:
            text += "".join(str(i)) + "\n"
        # text += '\n<hr/>\n'
    return text


# -----------------------------------------------------------------------------
# define_env
# contains the macros definitions


def define_env(env):
    """
    This is the hook for defining variables, macros and filters

    - variables: the dictionary that contains the environment variables
    - macro: a decorator function, to declare a macro.
    - filter: a function with one of more arguments,
        used to perform a transformation
    """

    @env.macro
    def implementations(module: str, function=None):
        return mod_to_string(module, function)

    @env.macro
    def bar(x):
        return (2.3 * x) + 7


if __name__ == "__main__":
    _ = mod_to_string("jwave.acoustics.time_varying", "simulate_wave_propagation")
    print(_)
