import pandas as pd 
import wandb

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

    for run in runs:
        name.append(run.name)
        tags.append(run.tags)
        state.append(run.state)
        url.append(run.url)
        created_at.append(run.created_at)
        path.append(run.path)
        notes.append(run.notes)

        df = pd.DataFrame([run.config], columns=run.config.keys())
        config_df = pd.concat([config_df, df], axis=0, ignore_index=True)
    col_names = ["run_name", "tags", "state", "url", "created_at", "path", "notes"]
    attrs_df = pd.DataFrame(zip(name, tags, state, url,created_at, path, notes), columns=col_names)

    combined_df =  pd.concat([config_df, attrs_df], axis=1)

    combined_df.insert(1,"run_name" , combined_df.pop("run_name"))
    combined_df.insert(3,"architecture" , combined_df.pop("architecture"))   
    combined_df.insert(2,"state" , combined_df.pop("state")) 
    combined_df.insert(4,"url" , combined_df.pop("url")) 
    
    return combined_df



def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'