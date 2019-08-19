import sys
import json
import os
import shutil
import errno
import subprocess
import utils


def copydir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        pass

json_file = sys.argv[1]

if len(sys.argv) == 3:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    import helpers
    hypersearcher = helpers.HyperSearcher()
    hypersearcher.run_hypersearch(json_file)
elif len(sys.argv) == 2:
    with open(json_file) as json_data:
        hyperconfig = json.load(json_data)
    script_dir = os.path.realpath('') + '/'
    script_copy_dir = hyperconfig['output_params']['hyper_dir'] + 'code/'
    copydir(script_dir, script_copy_dir)
    p1 = subprocess.Popen(['python', 'mibi_pickle.py', json_file])
    p1.communicate()
    utils.log_process(p1, hyperconfig['output_params']['hyper_dir'] + 'process_log.txt')

    p2 = subprocess.Popen(['python', script_copy_dir + 'hypersearch.py', json_file, 'copy'])
    p2.communicate()
    utils.log_process(p2, hyperconfig['output_params']['hyper_dir'] + 'process_log.txt')
else:
    print('Incorrect number of arguments passed')