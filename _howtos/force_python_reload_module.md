---
title: "How to force Python to reload a module"
---
Once a module has been loaded using `import module_name`, running this same command again will not reload the module. 

Say you are making changes on a module and testing the result interactively in a python shell. If you have loaded the module once and want to see the new changes you have to use:

```{python}
import importlib
importlib.reload(module_name)
