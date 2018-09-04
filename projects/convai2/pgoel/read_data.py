from parlai.tasks.convai2.agents import SelfOriginalTeacher
from parlai.scripts.train_model import setup_args

import pickle

parser = setup_args()
parser.set_defaults(datatype='test')
opt = parser.parse_args()
with open('model_file.opt', 'wb') as handle:
    pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

t = SelfOriginalTeacher(opt)
print("Number of episodes: ", t.num_episodes())
with open('self_original_test_simplified', 'w') as f:
    for i in range(t.num_episodes()):
        done = False
        j = 0
        while not done:
            table = t.data.get(i, j)
            input_utt = table[0]['text']
            output_utt = table[0]['labels'][0]
            f.write(input_utt + "\n")
            f.write(output_utt + "\n")
            done = table[0]['episode_done'] 
            j += 1
        f.write("\n")
