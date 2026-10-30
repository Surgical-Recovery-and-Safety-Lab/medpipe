from pydantic import ValidationError

from medpipe import MedpipeRegressor

try:
    pipe = MedpipeRegressor("default_regressor_config.toml")
    pipe.run()
except ValidationError as err:
    print(err)
    exit()
