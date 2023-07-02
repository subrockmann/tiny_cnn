import pandas as pd 
import json
import wandb

ENTITY = "susbrock"
PROJECT = "model_DB"

def wandb_model_DB():
    api = wandb.Api()
    runs = api.runs("susbrock/model_DB")

    config_df = pd.DataFrame()
    name = []
    tags =[]
    state = []
    url = []
    created_at = []
    path = []
    notes = []
    test_accuracies = []
    inference_avg_us = []
    stm32_inference_ms_INT8 =[]

    for run in runs:
        name.append(run.name)
        tags.append(run.tags)
        state.append(run.state)
        url.append(run.url)
        created_at.append(run.created_at)
        path.append(run.path)
        notes.append(run.notes)
        test_accuracies.append(run.summary.get("test_accuracy"))
        try:
            inference_avg_us.append(run.summary.get("inference_avg_us"))
        except:
            inference_avg_us.append(0)
        try:
            stm32_inference_ms_INT8.append(run.summary.get("stm32_inference_ms_INT8"))
        except:
            stm32_inference_ms_INT8.append(0)


        df = pd.DataFrame([run.config], columns=run.config.keys())
        config_df = pd.concat([config_df, df], axis=0, ignore_index=True)
    col_names = ["run_name", "tags", "state", "url", "created_at", "path", "notes", "test_accuracy", "inference_avg_us", "stm32_inference_ms_INT8"]
    attrs_df = pd.DataFrame(zip(name, tags, state, url,created_at, path, notes, test_accuracies, inference_avg_us, stm32_inference_ms_INT8), columns=col_names)

    combined_df =  pd.concat([config_df, attrs_df], axis=1)

    combined_df.insert(1,"run_name" , combined_df.pop("run_name"))
    combined_df.insert(3,"architecture" , combined_df.pop("architecture"))   
    combined_df.insert(2,"state" , combined_df.pop("state")) 
    combined_df.insert(4,"url" , combined_df.pop("url")) 
    
    return combined_df

def get_model_DB_run_id_from_architecture(architecture):
    df  = wandb_model_DB()

    run_id = df.query(f"architecture=='{architecture}'")["id"].values[0]
    return run_id

def get_architecture_from_model_DB_run_id(run_id):
    # get model name from run_id
    df  = wandb_model_DB()
    model_name = df.query(f"id=='{run_id}'")['architecture'].values[0]
    return model_name

def wandb_vww_training_df():
    api = wandb.Api()
    runs = api.runs("susbrock/model_DB_visual_wake_words")

    config_df = pd.DataFrame()
    name = []
    tags =[]
    state = []
    url = []
    created_at = []
    path = []
    notes = []
    test_accuracies = []
    #test_losses = []

    for run in runs:
        name.append(run.name)
        tags.append(run.tags)
        state.append(run.state)
        url.append(run.url)
        created_at.append(run.created_at)
        path.append(run.path)
        notes.append(run.notes)
        test_accuracies.append(run.summary.get("test_accuracy"))

        run_df = pd.DataFrame([run.config], columns=run.config.keys())
        config_df = pd.concat([config_df, run_df], axis=0, ignore_index=True)
    col_names = ["run_name", "tags", "state", "url", "created_at", "path", "notes", "test_accuracy"]
    attrs_df = pd.DataFrame(zip(name, tags, state, url,created_at, path, notes, test_accuracies), columns=col_names)

    training_df =  pd.concat([config_df, attrs_df], axis=1)

    training_df.insert(1,"run_name" , training_df.pop("run_name"))
    training_df.insert(3,"architecture" , training_df.pop("architecture"))   
    training_df.insert(2,"state" , training_df.pop("state")) 
    training_df.insert(4,"url" , training_df.pop("url")) 

    return training_df #, run

def get_vww_training_run_id_from_architecture(architecture):
    df  = wandb_vww_training_df()

    run_id = df.query(f"architecture=='{architecture}'")["id"].values[0]
    return run_id


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'

def color_state(val):
    color =''
    if val == "finished":
        color = 'green'
    elif val == "crashed":
        color = 'red'
    return 'background-color: %s' % color

def get_wandb_table_as_df(run_id, table_name, entity=ENTITY, project=PROJECT):
    """Retrieve a logged table from wandb and return it as a pandas dataframe

    Args:
        run_id (str): wandb run id
        table_name (str): name of the logged table
        entity (str, optional): name of the wandb workspace. Defaults to ENTITY.
        project (str, optional): name of the wandb project. Defaults to PROJECT.

    Returns:
        pd.DataFrame: pandas dataframe of the logged table
    """
    api = wandb.Api()
    artifact = api.artifact(f'{entity}/{project}/run-{run_id}-{table_name}:latest')
    artifact_dir = artifact.download()
    table_path = f"{artifact_dir}/{table_name}.table.json"
    print(f"filepath {table_path}")
    with open(table_path) as file:
        json_dict = json.load(file)
    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])

    return df
