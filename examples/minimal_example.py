from pydantic import ValidationError

from medpipe import MedpipeClassifier

try:
    pipe = MedpipeClassifier("default_classifier_config.toml")
    pipe.run()
except ValidationError as err:
    print(err)
    exit()
