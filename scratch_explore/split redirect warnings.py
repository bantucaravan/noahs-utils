old_showwarning = warnings.showwarning

def new_showwarning(*args, flush=True):
    old_showwarning(*args)
    file_handler = [i for i in logging.getLogger().handlers if isinstance(i,logging.FileHandler)][0]
    # NB: all args are passed to show warning from warnings.warn() as positional
    assert len(args) == 6, 'Expecting all args as positional'
    args = list(args)
    # file= is in the 5th position
    args[4] = file_handler.stream
    old_showwarning(*args)
    if flush:
        file_handler.stream.flush() # there seemed to be lag in printing

warnings.showwarning = new_showwarning
# CODE # warnings.warn('lol')
warnings.showwarning = old_showwarning

