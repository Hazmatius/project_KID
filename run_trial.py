import json
import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from . import helpers
import helpers

if sys.argv[1] == 'train':
    hyper_config = json.loads(sys.argv[2])
    trial_num = sys.argv[3]

    trial = helpers.Trial(hyper_config, trial_num)
    trial.train()
    trial.test()
    trial.save()
    sys.exit(0)

elif sys.argv[1] == 'check':
    hyper_config = json.loads(sys.argv[2])
    train_ds_path = sys.argv[3]

    check = helpers.Trial.error_check(hyper_config, train_ds_path)
    if check:
        sys.exit(0)
    else:
        sys.exit(1)