'''
Load a pickled object using namespace of different module than the 
module the obj was first pickled with reference to i.e. reassign module from 
which to load objects.
'''
import pickle

# from code based on https://stackoverflow.com/a/40916570/9426242 already in notes
class ModuleSwapUnpickler(pickle.Unpickler):
    #TODO: turn old_module, new_module into a dict to accomadate multiple swaps
    def __init__(self, file_obj, old_module, new_module, *args, **kwargs):
        '''
        Loads a pickled object with functions/classes imported from alternate 
        module as specified. Useful if you want to unpicle a user defined 
        object, but the use has changed the module name since the object was 
        pickled.
        
        old_module and new_module are strings.
        Module names are the full "path" used import the module in an import statement.
        '''
        self.new_module = new_module
        self.old_module = old_module
        self.original_modules = set()
        self.missing_modules = set()
        super().__init__(file_obj, *args, **kwargs)
    
    def find_class(self, module, name):
        '''
        The original pickled module will be passed as the module arg.
        Objects from unchanged modules get imported normally.
        '''
        #print(module)
        self.original_modules.add(module)
        #print(module)
        if module == self.old_module:
            module = self.new_module
            print(f'module "{self.old_module}" replaced by module "{self.new_module}" as source of object "{name}"')
        try:
            obj = super().find_class(module, name)
        except ModuleNotFoundError as e:
            if e.name in module:
                self.missing_modules.add(module)
            else:
                self.missing_modules.add(f'{e.name} while importing {module}')
            obj = type(name, (), {})
        return obj
    
    def load(self):
        '''Warn if old_module was not called'''
        out = super().load()
        if len(self.missing_modules) > 0:  
            raise ModuleNotFoundError(f'The following modules were not found: {self.missing_modules}')
        if self.old_module and (not self.old_module in self.original_modules):
            print(f'old_module "{self.old_module}" was not requested in reconstructing the object {type(out)}')
            print('pickle.load() requested these modules:', self.original_modules)
        return out


def pickle_module_swap(pkl_file, old_module=None, new_module=None, new_pkl_file=None):
    '''
    Save (or overwrite) pickled obj with reference to new module name from 
    which to load.

    To list the names of modules called by pickle load but not 
    availble in the current python env, set old_module=None (new_module can be 
    anything) and the unavailable modules (if any) will be returned as a set.

    NB: if you originally pickle dumped with a lower protocol, and you choose 
    to overwrite the pickle file, the lower protocol will be lost and the 
    highest protocol used.
    '''
    with open(pkl_file, 'rb') as f:
        # obj loaded from new_module
        module_swapper = ModuleSwapUnpickler(f, old_module, new_module)
        try:
            obj = module_swapper.load()
        except ModuleNotFoundError as e:
            print(e)
            return module_swapper.missing_modules
    #print(type(obj))
    pkl_file = new_pkl_file if new_pkl_file else pkl_file
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f, -1)



if __name__ == "__main__":
    # %load_ext autoreload
    # %autoreload 2

    #### example
    from noahs.general_utils import load_pickle
    file = '../Saved data pipelines/run_id_49-feature_selector.pkl'
    newfile='../Saved Models/test.pkl'
    unavailable_modules = pickle_module_swap(file) # discover any unavailable modules
    old_module='sklearn.decomposition._pca'
    new_module = 'proj_lib.custom_objects'
    pickle_module_swap(file, new_pkl_file=newfile, old_module=old_module,
        new_module=new_module)

    # check module has changed
    type(load_pickle(newfile))

    # overwrite original file
    pickle_module_swap(file, old_module=old_module, new_module=new_module)




    # example 2
    from noahs.general_utils import load_pickle
    file = "../Saved Models/model_stacked_runids['4516', '7262', '8280']_high_delta_avg-prec75.13_2021-01-08_05h46m21s.pkl"
    newfile='../Saved Models/test.pkl'
    unavailable_modules = pickle_module_swap(file) # discover any unavailable modules
    old_module='proj_lib.custom_objects'
    new_module = 'lib.custom_objects'
    pickle_module_swap(file, newfile=newfile, old_module=old_module,
        new_module=new_module)


    type(load_pickle(file))


