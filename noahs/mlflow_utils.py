# for any obj I would want converted back into python to reconstruct the pipeline... maybe 
# just save whole obj as pkl. Would that be a huge space suck if the goal is just worst 
# case exact flow reprodicibility... (consider pruning low scoring flows...)


def set_artifact(mlflow, value, key=None):
    '''
    Params:
        value: python obj to save
    '''
    # tempfile module?
    dir = mlflow.get_artifact_uri()[len('file://'):]
    dir = requests.utils.unquote(dir)
    if not key:
        key = np.random.rand(1)[0]
    # why not just save to "artifacts" folder if I know where it is
    path=os.path.join(dir,f'{key}.pkl')
    save_pickle(value, path)
    
    
def get_artifact(mlflow, run_id, key):
    ### fix based on set_artifact above!!!!!!!
    with mlflow.start_run(run_id): # check run_id is not currently active?
        uri = mlflow.get_artifact_uri(artifact_path=key)
        obj = load_pickle(uri) # get rid of ://file...?
    return obj



def extend_mlflow(mlflow):
# Other ways to get mlflow from the global space into which these funcs
# are imported into this funcs?
    mlflow.set_artifact = set_artifact
    mlflow.get_artifact = get_artifact
    return mlflow