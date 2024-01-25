## Get Started

1. Create env: install [go-task](https://taskfile.dev/) with `mamba install go-task`. To create env, run `task create-env`. Make sure it is activated.
2. Create new kaggle token from https://www.kaggle.com/settings/account
3. Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.
4. Run chmod 600 ~/.kaggle/kaggle.json
5. Get competition data with `task prepare-competition-data`
6. Run notebooks by order


## Goals

- [ ] Seek for more data
- [ ] Create a data generator
- [ ] Generate data
- [ ] Gain a 100% recall classifier with not too low precision.
- [ ] Train LLM classifier over the classified tokens.
