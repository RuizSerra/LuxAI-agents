from IPython.core.magic import register_cell_magic

# iPython magic to both write to file and execute cell
# https://stackoverflow.com/questions/33358611/ipython-notebook-writefile-and-execute-cell-at-the-same-time
@register_cell_magic
def write_and_run(line, cell):
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)


def obj_dir(obj):
    """
    >>> obj_dir(game_state.players[0])
    """
    print(type(obj))
    print()
    for a in dir(obj):
        if not a.startswith('__'):
            print(a, getattr(obj, a))
