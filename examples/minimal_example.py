from pydantic import ValidationError

from medpipe import Medpipe

try:
    pipe = Medpipe("default_config.toml")
    pipe.run()
except ValidationError as err:
    print(err)
    exit()
